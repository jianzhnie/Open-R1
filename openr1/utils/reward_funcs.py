import json
import math
import re
from typing import Dict, List, Set, Tuple, Union

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from transformers.utils.import_utils import _is_package_available

# Use same as transformers.utils.import_utils
_e2b_available = _is_package_available('e2b')


def is_e2b_available() -> bool:
    return _e2b_available


if is_e2b_available():
    from dotenv import load_dotenv
    from e2b_code_interpreter import Sandbox

    load_dotenv()


class BaseRewardFunction:
    """Placeholder for the base reward function class."""

    def __init__(self, **kwargs):
        self.config = kwargs

    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        raise NotImplementedError

    def validate_input(self, completions, solution=None):
        """统一的输入验证."""
        pass


class MathAccuracyReward(BaseRewardFunction):
    """Computes a reward based on whether the model's response is
    mathematically equivalent to the ground truth solution using latex2sympy2
    and math_verify.

    **Reward Criteria:**
        - ✅ 1.0 → If the response is **mathematically equivalent** to the solution.
        - ❌ 0.0 → If the response is **incorrect**.
        - 🔄 0.5 → If the **ground truth cannot be parsed**, to avoid unfair penalties.

    **Key Features:**
        - Parses mathematical expressions into symbolic form.
        - Compares model-generated answers with ground truth.
        - Handles edge cases where solutions are unparseable.

    **Args:**
        completions (List[str]): Model-generated completions.
        solution (List[str]): Ground truth solutions.

    **Returns:**
        List[float]: Reward scores between 0.0 and 1.0.
    """

    def __init__(self):
        """Initializes the MathAccuracyReward function with parsing
        configurations."""
        self.latex_config = [
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,  # Enforces correct LaTeX syntax
                    basic_latex=True,
                    equations=True,
                    boxed='all',
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ]

    def parse_expression(self, expression: str,
                         extraction_config: List[LatexExtractionConfig]):
        """Parses a mathematical expression using latex2sympy2.

        Args:
            expression (str): The input mathematical expression in LaTeX.

        Returns:
            Parsed expression object or None if parsing fails.
        """
        if not expression.strip():
            return None  # Avoid parsing empty strings

        try:
            return parse(expression,
                         extraction_mode='first_match',
                         extraction_config=extraction_config)
        except Exception as e:
            print(f'Parsing failed for expression: {expression}, Error: {e}')
            return None

    def __call__(self, completions: List[str], solution: List[str],
                 **kwargs) -> List[float]:
        """Computes accuracy-based rewards for mathematical expressions.

        Args:
            completions (List[str]): Model-generated responses.
            solution (List[str]): Ground truth solutions.

        Returns:
            List[float]: Rewards based on correctness.
        """
        rewards = []
        for content, sol in zip(completions, solution):
            gold_parsed = self.parse_expression(
                sol, extraction_config=[LatexExtractionConfig()])

            if gold_parsed is None:
                # Assign neutral reward if the ground truth cannot be parsed
                print(f'Warning: Failed to parse gold solution: {sol}')
                rewards.append(0.5)
                continue

            answer_parsed = self.parse_expression(
                content, extraction_config=self.latex_config)

            if answer_parsed is None:
                rewards.append(0.0)  # Invalid model response
                continue

            try:
                # If the verification function succeeds, return the verification score (1.0 or 0.0)
                reward = float(verify(answer_parsed, gold_parsed))
            except Exception as e:
                print(
                    f'Verification failed: {e}, Answer: {answer_parsed}, Gold: {gold_parsed}'
                )
                reward = 0.0

            rewards.append(reward)

        return rewards


class FormatReward(BaseRewardFunction):
    """Reward function to check if the completion follows the correct format:

    Expected format:
    ```
    <think>...</think> <answer>...</answer>
    ```

    - Matches two possible formats:
        1. Multi-line format:
            ```
            <think>
            ...
            </think>
            <answer>
            ...
            </answer>
            ```
        2. Single-line format:
            ```
            <think>...</think> <answer>...</answer>
            ```

    A reward of `1.0` is given for correctly formatted responses, otherwise `0.0`.
    """

    def __init__(self):
        # Regular expression patterns for checking format validity
        self.patterns = [
            re.compile(r'^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$',
                       re.DOTALL | re.MULTILINE),
            re.compile(r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])',
                       re.DOTALL | re.MULTILINE),
        ]

    def is_valid_format(self, content: str) -> bool:
        """Checks if the given content matches at least one of the valid
        formats.

        Args:
            content (str): The completion text to validate.

        Returns:
            bool: True if the format is valid, False otherwise.
        """
        return any(pattern.match(content) for pattern in self.patterns)

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """Evaluates whether each completion follows the expected format.

        Args:
            completions (List[str]): List of generated completions.

        Returns:
            List[float]: List of rewards (1.0 if formatted correctly, else 0.0).
        """
        return [
            1.0 if self.is_valid_format(content) else 0.0
            for content in completions
        ]


class ReActFormat(BaseRewardFunction):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific
        format."""
        pattern = r'^<think>.*?</think>\s*Action:.*?Action Input:.*?$'
        matches = [
            re.match(pattern, content, re.DOTALL | re.MULTILINE)
            for content in completions
        ]
        return [1.0 if match else 0.0 for match in matches]


class TagCountReward(BaseRewardFunction):
    """Reward function that checks if the generated text includes the correct
    number of `<think>` and `<answer>` tags, ensuring proper structure.

    - Each correctly placed tag contributes `0.25` to the reward.
    - Maximum reward = `1.0` (when all tags are correctly placed).

    Reference:
    Adapted from https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90

    Args:
        completions (List[str]): List of model-generated completions.

    Returns:
        List[float]: List of rewards, where `1.0` is the highest score.
    """

    @staticmethod
    def count_tags(text: str) -> float:
        """Counts the number of correctly placed `<think>` and `<answer>` tags.

        Args:
            text (str): The generated text to analyze.

        Returns:
            float: A reward score between `0.0` and `1.0`, based on tag correctness.
        """
        reward = 0.0
        if text.count('<think>\n') == 1:
            reward += 0.25
        if text.count('\n</think>\n') == 1:
            reward += 0.25
        if text.count('\n<answer>\n') == 1:
            reward += 0.25
        if text.count('\n</answer>') == 1:
            reward += 0.25
        return reward

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """Evaluates the presence of properly structured `<think>` and
        `<answer>` tags.

        Args:
            completions (List[str]): List of model-generated completions.

        Returns:
            List[float]: List of rewards, each ranging from `0.0` to `1.0`.
        """
        return [self.count_tags(content) for content in completions]


class ReasoningStepReward(BaseRewardFunction):
    r"""
    Reward function that checks for clear step-by-step reasoning.

    This function scans for structural elements indicating stepwise logical reasoning,
    such as "Step 1:", numbered lists, bullet points, and transition words.

    **Regex Pattern Matches:**
    - `Step \d+:` → Matches "Step 1:", "Step 2:", etc.
    - `^\d+\.` → Matches numbered lists like "1.", "2.", at the start of a line.
    - `\n-` → Matches bullet points using hyphens.
    - `\n\*` → Matches bullet points using asterisks.
    - `First,|Second,|Next,|Finally,` → Matches common transition words.

    **Reward Calculation:**
    - The function counts the number of reasoning step indicators in each completion.
    - A target of **3 or more indicators** yields the maximum reward (`1.0`).
    - If fewer than 3 indicators are found, a **proportional reward** (`count / 3`) is assigned.

    **Example Rewards:**
    - 3 or more indicators → `1.0`
    - 2 indicators → `0.67`
    - 1 indicator → `0.33`
    - 0 indicators → `0.0`
    """

    # Regex pattern to detect reasoning step indicators
    REASONING_PATTERN = re.compile(
        r'(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)',
        re.MULTILINE)

    @staticmethod
    def count_reasoning_steps(text: str) -> int:
        """Counts occurrences of reasoning step indicators in the given text.

        Args:
            text (str): The generated completion text.

        Returns:
            int: Number of matched reasoning steps.
        """
        return len(ReasoningStepReward.REASONING_PATTERN.findall(text))

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """Computes rewards based on the presence of reasoning step indicators.

        Args:
            completions (List[str]): List of generated text completions.

        Returns:
            List[float]: Reward scores between `0.0` and `1.0` for each completion.
        """
        return [
            min(1.0,
                self.count_reasoning_steps(content) / 3)
            for content in completions
        ]


class LengthReward(BaseRewardFunction):
    """Computes length-based rewards to discourage overthinking and promote
    token efficiency.

    Reference: Kimi 1.5 tech report (https://arxiv.org/abs/2501.12599)

    This function balances correctness and response length, rewarding concise correct answers.

    Reward Calculation:
    - Correct answers: `reward = 0.5 - (length - min_length) / (max_length - min_length)`
    - Incorrect answers: `reward = min(0, 0.5 - (length - min_length) / (max_length - min_length))`

    Args:
        completions (List[Dict[str, str]]): List of model-generated completions.
        solution (List[str]): List of ground truth solutions.

    Returns:
        List[float]: List of computed rewards for each completion.
    """

    def check_correctness(self, completions: List[Dict[str, str]],
                          solutions: List[str]) -> List[bool]:
        """Checks the correctness of each completion by comparing it to the
        ground truth solution.

        Args:
            completions (List[Dict[str, str]]): Generated completions.
            solutions (List[str]): Ground truth solutions.

        Returns:
            List[bool]: A list indicating whether each completion is correct.
        """
        correctness = []
        for content, sol in zip(completions, solutions):
            gold_parsed = parse(
                sol,
                extraction_mode='first_match',
                extraction_config=[LatexExtractionConfig()],
            )

            if not gold_parsed:
                # Treat as correct to avoid penalization when parsing fails
                correctness.append(True)
                print(f'Failed to parse gold solution: {sol}')
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode='first_match',
            )

            correctness.append(verify(answer_parsed, gold_parsed))

        return correctness

    def __call__(self, completions: List[Dict[str, str]], solution: List[str],
                 **kwargs) -> List[float]:
        """Computes length-based rewards for each completion.

        Args:
            completions (List[Dict[str, str]]): List of model-generated completions.
            solution (List[str]): List of ground truth solutions.

        Returns:
            List[float]: List of computed rewards.
        """
        correctness = self.check_correctness(completions, solution)

        # Calculate response lengths
        lengths = [len(content) for content in completions]
        min_len, max_len = min(lengths), max(lengths)

        # If all responses are of equal length, return zero rewards
        if max_len == min_len:
            return [0.0] * len(completions)

        rewards = []
        for length, is_correct in zip(lengths, correctness):
            lambda_val = 0.5 - (length - min_len) / (max_len - min_len)
            reward = lambda_val if is_correct else min(0, lambda_val)
            rewards.append(float(reward))

        return rewards


class CosineScaledReward(BaseRewardFunction):
    """Reward function that scales based on completion length using a cosine
    schedule.

    Reference: https://arxiv.org/abs/2502.03373

    Shorter correct completions receive higher rewards.
    Longer incorrect completions receive lower penalties.

    Args:
        cosine_min_value_wrong (float): Minimum reward for incorrect answers.
        cosine_max_value_wrong (float): Maximum reward for incorrect answers.
        cosine_min_value_correct (float): Minimum reward for correct answers.
        cosine_max_value_correct (float): Maximum reward for correct answers.
        cosine_max_len (int): Maximum length for scaling.
        accuracy_orm (BaseRewardFunction, optional): Accuracy computation module.

    Example:
        >>> reward_fn = CosineScaledReward()
        >>> rewards = reward_fn(["答案是42", "错误答案"], ["答案是42", "答案是43"])
        >>> print(rewards)
    """

    def __init__(
        self,
        cosine_min_value_wrong: float = -1.0,
        cosine_max_value_wrong: float = -0.5,
        cosine_min_value_correct: float = 0.5,
        cosine_max_value_correct: float = 1.0,
        cosine_max_len: int = 1000,
        accuracy_orm: Union[BaseRewardFunction, None] = None,
    ):
        self.min_value_wrong = cosine_min_value_wrong
        self.max_value_wrong = cosine_max_value_wrong
        self.min_value_correct = cosine_min_value_correct
        self.max_value_correct = cosine_max_value_correct
        self.max_len = cosine_max_len
        self.accuracy_orm = accuracy_orm or MathAccuracyReward()

    @staticmethod
    def cosine_scaled_reward(t: int, T: int, min_value: float,
                             max_value: float) -> float:
        """Computes a cosine-scaled reward value based on response length.

        Args:
            t (int): Current length of the completion.
            T (int): Maximum length for scaling.
            min_value (float): Minimum reward value.
            max_value (float): Maximum reward value.

        Returns:
            float: Scaled reward value.
        """
        cosine_value = math.cos(t * math.pi / T)
        return min_value + 0.5 * (max_value - min_value) * (1.0 + cosine_value)

    def __call__(self, completions: List[str], solution: List[str],
                 **kwargs) -> List[float]:
        """Computes cosine-scaled rewards for a list of model completions.

        Args:
            completions (List[str]): List of generated completions.
            solution (List[str]): List of ground truth solutions.
            **kwargs: Additional arguments for the accuracy function.

        Returns:
            List[float]: List of computed rewards.
        """
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        rewards = []

        for content, acc_reward in zip(completions, acc_rewards):
            is_correct = acc_reward >= 1.0
            min_value = self.min_value_correct if is_correct else self.max_value_wrong
            max_value = self.max_value_correct if is_correct else self.min_value_wrong
            gen_text_len = len(content)
            reward = cosine_scaled_reward(gen_text_len, self.max_len,
                                          min_value, max_value)
            rewards.append(reward)

        return rewards


class RepetitionPenalty(BaseRewardFunction):
    """Computes an N-gram repetition penalty as described in Appendix C.2 of:
    https://arxiv.org/abs/2502.03373.

    This function penalizes responses that contain repeated sequences of `n` words.

    **Reference Implementation:**
    - Adapted from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    **Reward Calculation:**
    - If an `n`-gram appears more than once, it contributes to a repetition penalty.
    - The penalty is scaled based on the proportion of repeated n-grams.
    - The maximum penalty (`max_penalty`) is applied when all n-grams are repeated.

    **Args:**
        repetition_n_grams (int): The size of the n-grams to check for repetition.
        repetition_max_penalty (float): The maximum negative penalty for highly repetitive text.

    **Raises:**
        ValueError: If `repetition_max_penalty` is positive.

    **Example Usage:**
        ```python
        penalty_fn = RepetitionPenalty(repetition_n_grams=3, repetition_max_penalty=-1.0)
        rewards = penalty_fn(["The cat sat on the mat. The cat sat on the mat."])
        ```
    """

    def __init__(self,
                 repetition_n_grams: int = 3,
                 repetition_max_penalty: float = -1.0):
        if repetition_max_penalty > 0:
            raise ValueError(
                f'`repetition_max_penalty` should not be positive: {repetition_max_penalty}'
            )

        self.ngram_size = repetition_n_grams
        self.max_penalty = repetition_max_penalty

    @staticmethod
    def extract_ngrams(text: str, ngram_size: int) -> Set[Tuple[str, ...]]:
        """Extracts all n-grams from the given text.

        Args:
            text (str): The input text.
            ngram_size (int): The size of n-grams to extract.

        Returns:
            Set[Tuple[str, ...]]: A set of unique n-grams found in the text.
        """
        words = text.lower().split()
        if len(words) < ngram_size:
            return set()
        return set(
            tuple(words[i:i + ngram_size])
            for i in range(len(words) - ngram_size + 1))

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """Computes repetition penalties for a list of completions.

        Args:
            completions (List[str]): List of generated text completions.

        Returns:
            List[float]: Penalty scores (negative values), with `0.0` meaning no penalty.
        """
        rewards = []
        for completion in completions:
            words = completion.split()

            # Handle empty or short completions (no penalty)
            if not completion or len(words) < self.ngram_size:
                rewards.append(0.0)
                continue

            # Compute total n-grams and unique n-grams
            total_ngrams = len(words) - self.ngram_size + 1
            unique_ngrams = len(
                self.extract_ngrams(completion, self.ngram_size))

            # Compute repetition scaling factor
            repetition_ratio = 1 - (unique_ngrams / total_ngrams)
            reward = repetition_ratio * self.max_penalty

            rewards.append(reward)

        return rewards


def extract_code(completion: str) -> str:
    pattern = re.compile(r'```python\n(.*?)```', re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ''
    return extracted_answer


class CodeReward(BaseRewardFunction):
    """Reward function that evaluates code snippets using the E2B code
    interpreter.

    Assumes the dataset contains a `verification_info` column with test cases.
    """

    def __call__(self, completions, solution, **kwargs) -> List[float]:

        if not is_e2b_available():
            raise ImportError(
                'E2B is not available and required for this reward function. Please install E2B with '
                '`pip install e2b-code-interpreter` and add an API key to a `.env` file.'
            )

        rewards = []
        # TODO: add support for other languages in E2B: https://e2b.dev/docs/code-interpreting/supported-languages
        try:
            """Returns a reward function that evaluates code snippets in a
            sandbox."""
            evaluation_script_template = """
            import subprocess
            import json

            def evaluate_code(code, test_cases):
                passed = 0
                total = len(test_cases)
                exec_timeout = 5

                for case in test_cases:
                    process = subprocess.run(
                        ["python3", "-c", code],
                        input=case["input"],
                        text=True,
                        capture_output=True,
                        timeout=exec_timeout
                    )

                    if process.returncode != 0:  # Error in execution
                        continue

                    output = process.stdout.strip()
                    if output.strip() == case["output"].strip():
                        passed += 1

                success_rate = (passed / total)
                return success_rate

            code_snippet = {code}
            test_cases = json.loads({test_cases})

            evaluate_code(code_snippet, test_cases)
            """
            verification_info = kwargs['verification_info']
            scripts = [
                evaluation_script_template.format(
                    code=json.dumps(code),
                    test_cases=json.dumps(json.dumps(info['test_cases'])))
                for code, info in zip(completions, verification_info)
            ]
            with Sandbox(timeout=30, request_timeout=3) as sbx:
                for script in scripts:
                    execution = sbx.run_code(
                        script, language=verification_info['language'])
                    try:
                        output = float(execution.text)
                    except (TypeError, ValueError):
                        output = 0.0
                    rewards.append(output)
        except Exception as e:
            print(f'Error from E2B executor: {e}')
            rewards = [0.0] * len(completions)
        return rewards


relu_based_reward_func_mapping = {
    'accuracy': MathAccuracyReward,
    'format': FormatReward,
    'react_format': ReActFormat,
    'tag_reward': TagCountReward,
    'reasoning_steps': ReasoningStepReward,
    'length': LengthReward,
    'cosine': CosineScaledReward,
    'repetition': RepetitionPenalty,
}


def cosine_scaled_reward(t: int, T: int, min_value: float,
                         max_value: float) -> float:
    """Computes a cosine-scaled reward value based on length.

    :param t: Current length of the response.
    :param T: Maximum length considered.
    :param min_value: Minimum reward value.
    :param max_value: Maximum reward value.
    :return: Scaled reward value.
    """
    cosine_value = math.cos(t * math.pi / T)
    return min_value + 0.5 * (max_value - min_value) * (1.0 + cosine_value)


def test_cosine_scaled_reward_behavior() -> None:
    """Tests the cosine_scaled_reward function for correct and incorrect
    answers with varying lengths, ensuring expected reward behavior."""
    # Test cases: correct answers (varying lengths)
    correct_answers = [
        '答案是42',  # Length: 5
        '经过仔细计算，答案是42',  # Length: 12
        '让我详细解释一下，经过认真思考和计算，最终答案是42'  # Length: 25
    ]

    # Test cases: incorrect answers (varying lengths)
    wrong_answers = [
        '答案是24',  # Length: 5
        '经过仔细计算，答案是24',  # Length: 12
        '让我详细解释一下，经过认真思考和计算，最终答案是24'  # Length: 25
    ]

    # Ground truth correctness (1 = correct, 0 = incorrect)
    accuracy_rewards: List[int] = [1, 1, 1, 0, 0, 0]
    all_contents = correct_answers + wrong_answers

    # Reward scaling parameters
    min_value_wrong, max_value_wrong = -1.0, -0.5
    min_value_correct, max_value_correct = 0.5, 1.0
    max_len = 20

    # Compute cosine-scaled rewards
    cosine_rewards = []
    for content, acc_reward in zip(all_contents, accuracy_rewards):
        gen_len = len(content)
        is_correct = acc_reward >= 1.0

        if is_correct:
            min_value = min_value_correct
            max_value = max_value_correct
        else:
            # Swap min/max for incorrect answers
            min_value = max_value_wrong
            max_value = min_value_wrong

        reward = cosine_scaled_reward(gen_len, max_len, min_value, max_value)
        cosine_rewards.append(reward)

    for (content, reward) in zip(all_contents, cosine_rewards):
        print(f'content: {content}, reward: {reward}')

    # Assertions to verify expected reward behavior
    assert cosine_rewards[0] > cosine_rewards[1] > cosine_rewards[
        2], 'Correct answers should receive decreasing rewards as length increases.'
    assert cosine_rewards[3] < cosine_rewards[4] < cosine_rewards[
        5], 'Incorrect answers should receive decreasing penalties as length increases.'

    print('All tests passed successfully!')


if __name__ == '__main__':
    test_cosine_scaled_reward_behavior()
