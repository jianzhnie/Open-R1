import json
import math
import os
import re
import sys
from typing import Callable, Dict, List, Sequence, Set, Tuple, Union

from latex2sympy2_extended import NormalizationConfig
from math_verify.grader import verify
from math_verify.parser import (ExprExtractionConfig, LatexExtractionConfig,
                                parse)
from transformers.utils.import_utils import _is_package_available

sys.path.append(os.getcwd())

from openr1.utils.logger_utils import get_logger

# 创建 Logger
logger = get_logger(__name__)

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

    def __init__(self, gold_is_latex: bool = True):
        """Initializes the MathAccuracyReward function with parsing
        configurations."""
        self.gold_extration_config = [
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
            )]
        self.gold_is_latex = gold_is_latex

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
            logger.info(
                f'Parsing failed for expression: {expression}, Error: {e}')
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
                logger.info(f'Warning: Failed to parse gold solution: {sol}')
                rewards.append(0.5)
                continue

            answer_parsed = self.parse_expression(
                content, extraction_config=self.gold_extration_config)

            if answer_parsed is None:
                rewards.append(0.0)  # Invalid model response
                continue

            try:
                # If the verification function succeeds, return the verification score (1.0 or 0.0)
                reward = float(verify(answer_parsed, gold_parsed))
            except Exception as e:
                logger.info(
                    f'Verification failed: {e}, Answer: {answer_parsed}, Gold: {gold_parsed}'
                )
                reward = 0.0

            rewards.append(reward)

        return rewards


class MathAccuracyRewardV2(BaseRewardFunction):
    """Computes accuracy-based rewards for mathematical expressions using
    latex2sympy2.

    **Key Enhancements:**
        - Supports both **LaTeX** and **symbolic expressions**.
        - Uses an **aggregation function** to handle multiple parsing strategies.
        - Adds **robust error handling** and logging.

    **Reward Criteria:**
        - ✅ 1.0 → Response is mathematically equivalent to the solution.
        - ❌ 0.0 → Response is incorrect.
        - 🔄 0.5 → Ground truth cannot be parsed.

    **Args:**
        completions (List[str]): Model-generated completions.
        solution (List[str]): Ground truth solutions.

    **Returns:**
        List[float]: Reward scores between 0.0 and 1.0.
    """

    def __init__(self, gold_is_latex: bool = True, **kwargs):
        super().__init__(**kwargs)

        # Ensure extraction config is a list (not a tuple)
        self.gold_extraction_config: Sequence = ([
            LatexExtractionConfig()
        ] if gold_is_latex else [ExprExtractionConfig()])
        self.pred_extraction_config: Sequence = [
            ExprExtractionConfig(),
            LatexExtractionConfig()
        ]

        self.aggregation_function: Callable[[list[float]], float] = max
        self.precision: int = 6

    def parse_expression(self, expression: str, extraction_config: Sequence):
        """Parses a mathematical expression using latex2sympy2.

        Args:
            expression (str): The input mathematical expression.

        Returns:
            Parsed expression object or None if parsing fails.
        """
        if not expression.strip():
            return None  # Skip empty expressions

        try:
            return parse(expression, extraction_config=extraction_config)
        except Exception as e:
            logger.warning(
                f'Parsing failed for expression: {expression}, Error: {e}')
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
                sol, extraction_config=self.gold_extraction_config)

            if not gold_parsed:
                # Assign neutral reward if the ground truth cannot be parsed
                logger.warning(f'Failed to parse ground truth solution: {sol}')
                rewards.append(0.5)
                continue

            answer_parsed = self.parse_expression(
                content, extraction_config=self.pred_extraction_config)

            if not answer_parsed:
                # Penalize unparseable model outputs
                rewards.append(0.0)
                continue

            try:
                # Compute reward using the aggregation function
                reward = self.aggregation_function([
                    1.0 if any(
                        verify(gold, pred, self.precision)
                        for gold in gold_parsed) else 0.0
                    for pred in answer_parsed
                ])
            except Exception as e:
                logger.error(
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
                logger.info(f'Failed to parse gold solution: {sol}')
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
    """Reward function that scales rewards based on completion length using a
    cosine schedule.

    **Reference**: https://arxiv.org/abs/2502.03373

    **Key Behavior**:
        - ✅ Shorter **correct** completions receive **higher** rewards.
        - ❌ Longer **incorrect** completions receive **lower** penalties.

    **Args:**
        - `cosine_min_value_wrong` (float): Minimum reward for incorrect answers.
        - `cosine_max_value_wrong` (float): Maximum reward for incorrect answers.
        - `cosine_min_value_correct` (float): Minimum reward for correct answers.
        - `cosine_max_value_correct` (float): Maximum reward for correct answers.
        - `cosine_max_len` (int): Maximum length for scaling.
        - `accuracy_orm` (BaseRewardFunction, optional): Accuracy computation module.

    **Example Usage**:
        ```python
        reward_fn = CosineScaledReward()
        rewards = reward_fn(["答案是42", "错误答案"], ["答案是42", "答案是43"])
        print(rewards)
        ```
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
        self.accuracy_orm = accuracy_orm or MathAccuracyRewardV2()

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
            gen_text_len = len(
                content.strip())  # Avoid counting excess whitespace

            if gen_text_len == 0:
                logger.warning(f'Skipping empty completion: {content}')
                rewards.append(self.min_value_wrong
                               )  # Assign minimum penalty for empty responses
                continue

            is_correct = acc_reward >= 1.0

            # Correct answers get higher rewards for being concise
            if is_correct:
                min_value, max_value = self.min_value_correct, self.max_value_correct
            else:
                min_value, max_value = self.max_value_wrong, self.min_value_wrong  # Fixed logic

            # Compute scaled reward
            reward = self.cosine_scaled_reward(gen_text_len, self.max_len,
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
            logger.info(f'Error from E2B executor: {e}')
            rewards = [0.0] * len(completions)
        return rewards


relu_based_reward_func_mapping = {
    'accuracy_v1': MathAccuracyReward,
    'accuracy_v2': MathAccuracyRewardV2,
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
        logger.info(f'content: {content}, reward: {reward}')

    # Assertions to verify expected reward behavior
    assert cosine_rewards[0] > cosine_rewards[1] > cosine_rewards[
        2], 'Correct answers should receive decreasing rewards as length increases.'
    assert cosine_rewards[3] < cosine_rewards[4] < cosine_rewards[
        5], 'Incorrect answers should receive decreasing penalties as length increases.'

    logger.info('All tests passed successfully!')


def test_rewards_func_exam1() -> None:
    """Test the reward function with various math-related completion
    examples."""
    reward_fn = CosineScaledReward()

    # Test cases: pairs of (generated answer, expected solution)
    examples: List[Tuple[str, str]] = [
        ('so that x == 1 or x == 2, thus the result is $2*\pi*r$',
         '$2*\pi*r$'), ('The answer is $$\\sin(x)$$', '$$\\sin(x)$$'),
        ('After solving, we get $1/2$', '$1/2$'),
        ('The final result is $$(a + b)^2$$', '$(a + b)^2$'),
        ('Therefore, $$3!$$', '$3!$'),
        ('The point coordinates are $(1,2)$', '$(1,2)$')
    ]

    completions, solutions = zip(*examples)
    rewards = reward_fn(list(completions), list(solutions))

    logger.info('\nReward Examples:')
    for comp, solution, reward in zip(completions, solutions, rewards):
        logger.info(f'Answer: {comp}')
        logger.info(f'Gold: {solution}')
        logger.info(f'Length: {len(comp)}')
        logger.info(f'Reward: {reward:.3f}\n')


def test_rewards_func_exam2() -> None:
    """Test the reward function with different mathematical reasoning examples.

    Includes correct, incorrect, and partially correct responses.
    """
    examples = [
        # ✅ 正确示例（包含推理步骤）
        (r'首先，我们知道二次方程 $2x - 3 = 0$。\n'
         r'移项得到 $2x = 3$。\n'
         r'两边同时除以 2，得出 $x = \frac{3}{2}$。', r'x = \frac{3}{2}'),  # 完整推理 + 正确答案
        (r'根据爱因斯坦的质能方程：\n'
         r'$E = mc^2$。\n'
         r'其中，$m$ 代表质量，$c$ 代表光速。', r'E = mc^2'),  # 物理公式 + 解释
        (r'计算定积分 $\int_0^1 x^2 \,dx$。\n'
         r'首先，计算不定积分：$\int x^2 \,dx = \frac{x^3}{3}$。\n'
         r'然后代入上限 1 和下限 0，得到：\n'
         r'$\frac{1^3}{3} - \frac{0^3}{3} = \frac{1}{3}$。',
         r'\int_0^1 x^2 \,dx = \frac{1}{3}'),  # 清晰的积分推导

        # ❌ 错误示例（包含推理错误）
        (r'解方程 $2x - 3 = 0$。\n'
         r'移项得到 $2x = 3$。\n'
         r'然后两边同时除以 **3**，得出 $x = \frac{3}{3} = 1$。', r'x = \frac{3}{2}'
         ),  # 计算错误（除错数）
        (r'根据物理公式：$E = mc^3$。\n'
         r'但实际上光速的指数应为 2，因此正确公式是 $E = mc^2$。', r'E = mc^2'),  # 公式错误
        (r'计算定积分 $\int_0^1 x^2 \,dx$。\n'
         r'计算不定积分：$\int x^2 \,dx = \frac{x^3}{3}$。\n'
         r'然后代入上限 1 和下限 0，得到：\n'
         r'$\frac{1^3}{2} - \frac{0^3}{2} = \frac{1}{2}$。',
         r'\int_0^1 x^2 \,dx = \frac{1}{3}'),  # 积分结果错误

        # 🔄 部分正确（推理清晰但格式问题或等价表达）
        (r'解方程 $2x - 3 = 0$。\n'
         r'移项得到 $2x = 3$。\n'
         r'两边同时除以 2，得出 $x = 1.5$。', r'x = \frac{3}{2}'),  # 结果正确但不是 LaTeX 形式
        (r'级数求和结果如下：\n'
         r'$\frac{\pi^2}{6} = \sum_{n=1}^{\infty} \frac{1}{n^2}$。',
         r'\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}'),  # 数学等价但顺序不同
        (r'计算定积分 $\int_0^1 x^2 \,dx$。\n'
         r'计算不定积分：$\int x^2 \,dx = \frac{x^3}{3}$。\n'
         r'然后代入上下限，得到 $x^3/3$ 的变化量。\n'
         r'最终答案为 $\frac{2}{6}$。',
         r'\int_0^1 x^2 \,dx = \frac{1}{3}'),  # 结果正确但未化简
        (r'爱因斯坦公式 $E = c^2 m$ 适用于质量与能量的转换。', r'E = mc^2'),  # 变量顺序错误但等价
    ]

    reward_fn = CosineScaledReward(cosine_max_len=30)
    reward_fn = MathAccuracyReward()
    completions, solutions = zip(*examples)
    rewards = reward_fn(list(completions), list(solutions))

    logger.info('\nReward Examples:')
    for comp, solution, reward in zip(completions, solutions, rewards):
        logger.info(f'Answer: {comp}')
        logger.info(f'Gold: {solution}')
        logger.info(f'Length: {len(comp)}')
        logger.info(f'Reward: {reward:.3f}\n')


if __name__ == '__main__':
    test_cosine_scaled_reward_behavior()
    test_rewards_func_exam1()
    test_rewards_func_exam2()
