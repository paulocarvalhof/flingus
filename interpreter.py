#!/usr/bin/env python3
"""
Flingus: a small plain-English programming language interpreter.

Usage:
  python flingus.py path/to/program.flingus
  python flingus.py --demo
"""

from __future__ import annotations

import argparse
import ast
import operator
import re
import sys
import time
from dataclasses import dataclass
from typing import Any


class FlingusError(Exception):
    """A user-facing Flingus runtime or syntax error."""


@dataclass
class SourceLine:
    number: int
    indent: int
    text: str


class FlingusInterpreter:
    def __init__(self, input_func=input, output_func=print, sleep_func=time.sleep):
        self.memory: dict[str, Any] = {}
        self.input = input_func
        self.output = output_func
        self.sleep = sleep_func

    def run(self, code: str) -> None:
        lines = self.prepare_lines(code)
        program_lines = self.strip_program_markers(lines)
        self.execute_block(program_lines, 0, len(program_lines))

    def prepare_lines(self, code: str) -> list[SourceLine]:
        result: list[SourceLine] = []

        for number, raw_line in enumerate(code.splitlines(), start=1):
            if not raw_line.strip():
                continue

            text = raw_line.strip()
            if text.startswith("Note:") or text.startswith("Explanation:"):
                continue

            indent = len(raw_line) - len(raw_line.lstrip(" "))
            if "\t" in raw_line[:indent]:
                raise FlingusError(
                    f"Line {number}: Please use spaces for indentation, not tabs."
                )

            result.append(SourceLine(number, indent, text))

        return result

    def strip_program_markers(self, lines: list[SourceLine]) -> list[SourceLine]:
        if not lines:
            return []

        start = 0
        end = len(lines)

        if lines[0].text.startswith("BEGIN_PROGRAM"):
            start = 1

        if end > start and lines[end - 1].text == "END_PROGRAM":
            end -= 1

        return lines[start:end]

    def execute_block(self, lines: list[SourceLine], start: int, end: int) -> None:
        index = start
        while index < end:
            line = lines[index]

            if self.is_block_header(line.text):
                child_start, child_end = self.find_child_block(lines, index, end)
                index = self.execute_block_statement(lines, index, child_start, child_end, end)
                continue

            if line.text == "Otherwise:":
                return

            self.execute_line(line)
            index += 1

    def execute_block_statement(
        self,
        lines: list[SourceLine],
        index: int,
        child_start: int,
        child_end: int,
        scope_end: int,
    ) -> int:
        line = lines[index]
        text = line.text

        if text.startswith("Repeat "):
            count = self.parse_repeat_count(line)
            for _ in range(count):
                self.execute_block(lines, child_start, child_end)
            return child_end

        if text.startswith("For each "):
            item_name, list_name = self.parse_for_each(line)
            value = self.get_variable(list_name, line)
            if not isinstance(value, list):
                self.fail(line, f"`{list_name}` is not a list, so I cannot loop through it.")
            for item in list(value):
                self.memory[item_name] = item
                self.execute_block(lines, child_start, child_end)
            return child_end

        if text.startswith("While "):
            condition_text = self.strip_header(text, "While")
            guard = 0
            while self.evaluate_condition(condition_text, line):
                self.execute_block(lines, child_start, child_end)
                guard += 1
                if guard > 100_000:
                    self.fail(
                        line,
                        "This While loop has run too many times. Check that it eventually becomes false.",
                    )
            return child_end

        if text.startswith(("If ", "When ", "Provided that ")):
            otherwise_index = self.find_otherwise(lines, child_end, scope_end, line.indent)
            condition_text = self.parse_if_condition(text)
            if self.evaluate_condition(condition_text, line):
                self.execute_block(lines, child_start, child_end)
                if otherwise_index is not None:
                    _, other_end = self.find_child_block(lines, otherwise_index, scope_end)
                    return other_end
            elif otherwise_index is not None:
                other_start, other_end = self.find_child_block(lines, otherwise_index, scope_end)
                self.execute_block(lines, other_start, other_end)
                return other_end
            return child_end

        self.fail(line, f"I do not understand this block instruction: {text}")
        return child_end

    def execute_line(self, line: SourceLine) -> None:
        text = self.remove_final_period(line.text)

        handlers = [
            self.handle_store,
            self.handle_print,
            self.handle_input,
            self.handle_calculate,
            self.handle_add,
            self.handle_remove,
            self.handle_count,
            self.handle_wait,
        ]

        for handler in handlers:
            if handler(text, line):
                return

        self.fail(line, f"I do not understand this instruction: {line.text}")

    def handle_store(self, text: str, line: SourceLine) -> bool:
        match = re.fullmatch(r"(Store|Remember|Hold) (.+) (in|as) (`[^`]+`)", text)
        if not match:
            return False

        variable_name = self.parse_variable(match.group(4), line)
        self.memory[variable_name] = self.evaluate(match.group(2), line)
        return True

    def handle_print(self, text: str, line: SourceLine) -> bool:
        if not text.startswith("Print "):
            return False

        self.output(self.evaluate(text[len("Print ") :], line))
        return True

    def handle_input(self, text: str, line: SourceLine) -> bool:
        match = re.fullmatch(
            r'Ask the user "([^"]*)" and store the answer in (`[^`]+`)', text
        )
        if match:
            question = match.group(1)
            variable_name = self.parse_variable(match.group(2), line)
            self.memory[variable_name] = self.input(question + " ")
            return True

        match = re.fullmatch(
            r'Ask the user for a number with the question "([^"]*)" and store it in (`[^`]+`)',
            text,
        )
        if match:
            question = match.group(1)
            variable_name = self.parse_variable(match.group(2), line)
            answer = self.input(question + " ")
            try:
                self.memory[variable_name] = float(answer) if "." in answer else int(answer)
            except ValueError:
                self.fail(line, f"I expected a number, but the user entered {answer!r}.")
            return True

        return False

    def handle_calculate(self, text: str, line: SourceLine) -> bool:
        match = re.fullmatch(
            r"Calculate (.+) (plus|minus|times|divided by|remainder of) (.+) and store the result in (`[^`]+`)",
            text,
        )
        if not match:
            return False

        left = self.evaluate(match.group(1), line)
        phrase = match.group(2)
        right = self.evaluate(match.group(3), line)
        target = self.parse_variable(match.group(4), line)
        operations = {
            "plus": operator.add,
            "minus": operator.sub,
            "times": operator.mul,
            "divided by": operator.truediv,
            "remainder of": operator.mod,
        }

        try:
            self.memory[target] = operations[phrase](left, right)
        except Exception as exc:
            self.fail(line, f"I could not calculate that: {exc}")
        return True

    def handle_add(self, text: str, line: SourceLine) -> bool:
        match = re.fullmatch(r"Add (.+) to (`[^`]+`)", text)
        if not match:
            return False

        value = self.evaluate(match.group(1), line)
        list_name = self.parse_variable(match.group(2), line)
        if list_name not in self.memory:
            self.memory[list_name] = []
        if not isinstance(self.memory[list_name], list):
            self.fail(line, f"`{list_name}` is not a list, so I cannot add to it.")
        self.memory[list_name].append(value)
        return True

    def handle_remove(self, text: str, line: SourceLine) -> bool:
        match = re.fullmatch(r"Remove (.+) from (`[^`]+`)", text)
        if not match:
            return False

        value = self.evaluate(match.group(1), line)
        list_name = self.parse_variable(match.group(2), line)
        values = self.get_variable(list_name, line)
        if not isinstance(values, list):
            self.fail(line, f"`{list_name}` is not a list, so I cannot remove from it.")
        try:
            values.remove(value)
        except ValueError:
            self.fail(line, f"`{list_name}` does not contain {value!r}.")
        return True

    def handle_count(self, text: str, line: SourceLine) -> bool:
        match = re.fullmatch(
            r"Count the items in (`[^`]+`) and store the result in (`[^`]+`)", text
        )
        if not match:
            return False

        source_name = self.parse_variable(match.group(1), line)
        target_name = self.parse_variable(match.group(2), line)
        value = self.get_variable(source_name, line)
        if not isinstance(value, (list, str, dict)):
            self.fail(line, f"I cannot count the items in `{source_name}`.")
        self.memory[target_name] = len(value)
        return True

    def handle_wait(self, text: str, line: SourceLine) -> bool:
        match = re.fullmatch(r"Wait ([0-9]+(?:\.[0-9]+)?) seconds?", text)
        if not match:
            return False

        self.sleep(float(match.group(1)))
        return True

    def evaluate(self, text: str, line: SourceLine) -> Any:
        text = text.strip()

        if " followed by " in text:
            return "".join(str(self.evaluate(part, line)) for part in text.split(" followed by "))

        if re.fullmatch(r"`[^`]+`", text):
            return self.get_variable(self.parse_variable(text, line), line)

        if text == "true":
            return True
        if text == "false":
            return False
        if text == "nothing":
            return None

        if re.fullmatch(r"-?[0-9]+", text):
            return int(text)
        if re.fullmatch(r"-?[0-9]+\.[0-9]+", text):
            return float(text)

        try:
            return ast.literal_eval(text)
        except Exception:
            self.fail(
                line,
                f"I do not understand this value: {text}. Use text in quotes, a number, true, false, a list, or a `variable`.",
            )

    def evaluate_condition(self, text: str, line: SourceLine) -> bool:
        text = text.strip()

        variable_bool = re.fullmatch(r"(`[^`]+`) is (true|false)", text)
        if variable_bool:
            value = self.evaluate(variable_bool.group(1), line)
            expected = variable_bool.group(2) == "true"
            return bool(value) is expected

        comparisons = [
            ("does not contain", lambda left, right: right not in left),
            ("is exactly equal to", lambda left, right: left == right),
            ("is not equal to", lambda left, right: left != right),
            ("is greater than", lambda left, right: left > right),
            ("is less than", lambda left, right: left < right),
            ("is at least", lambda left, right: left >= right),
            ("is at most", lambda left, right: left <= right),
            ("contains", lambda left, right: right in left),
        ]

        for phrase, compare in comparisons:
            split = f" {phrase} "
            if split in text:
                left_text, right_text = text.split(split, 1)
                left = self.evaluate(left_text, line)
                right = self.evaluate(right_text, line)
                try:
                    return bool(compare(left, right))
                except Exception as exc:
                    self.fail(line, f"I could not compare those values: {exc}")

        self.fail(
            line,
            f"I do not understand this condition: {text}. Try something like `age` is greater than 18.",
        )
        return False

    def is_block_header(self, text: str) -> bool:
        return (
            text.startswith("Repeat ")
            or text.startswith("For each ")
            or text.startswith("While ")
            or text.startswith("If ")
            or text.startswith("When ")
            or text.startswith("Provided that ")
        ) and text.endswith(":")

    def find_child_block(
        self, lines: list[SourceLine], header_index: int, scope_end: int
    ) -> tuple[int, int]:
        header = lines[header_index]
        child_start = header_index + 1

        if child_start >= scope_end or lines[child_start].indent <= header.indent:
            self.fail(header, "This instruction needs indented lines under it.")

        child_indent = lines[child_start].indent
        child_end = child_start
        while child_end < scope_end:
            current = lines[child_end]
            if current.indent < child_indent:
                break
            if current.indent == header.indent and current.text == "Otherwise:":
                break
            child_end += 1

        return child_start, child_end

    def find_otherwise(
        self,
        lines: list[SourceLine],
        start: int,
        end: int,
        indent: int,
    ) -> int | None:
        if start < end and lines[start].indent == indent and lines[start].text == "Otherwise:":
            return start
        return None

    def parse_repeat_count(self, line: SourceLine) -> int:
        match = re.fullmatch(r"Repeat ([0-9]+) times:", line.text)
        if not match:
            self.fail(line, "To repeat actions, write: Repeat 3 times:")
        return int(match.group(1))

    def parse_for_each(self, line: SourceLine) -> tuple[str, str]:
        match = re.fullmatch(r"For each (`[^`]+`) in (`[^`]+`):", line.text)
        if not match:
            self.fail(line, "To loop through a list, write: For each `item` in `items`:")
        return self.parse_variable(match.group(1), line), self.parse_variable(match.group(2), line)

    def parse_if_condition(self, text: str) -> str:
        if text.startswith("Provided that "):
            return text[len("Provided that ") : -1]
        first_space = text.find(" ")
        return text[first_space + 1 : -1]

    def parse_variable(self, text: str, line: SourceLine) -> str:
        match = re.fullmatch(r"`([^`]+)`", text.strip())
        if not match:
            self.fail(line, f"Expected a variable name written with backticks, like `name`, but got {text}.")
        return match.group(1)

    def get_variable(self, name: str, line: SourceLine) -> Any:
        if name not in self.memory:
            self.fail(line, f"I do not remember a variable named `{name}`. Store something in it before using it.")
        return self.memory[name]

    def strip_header(self, text: str, word: str) -> str:
        return text[len(word) + 1 : -1]

    def remove_final_period(self, text: str) -> str:
        return text[:-1] if text.endswith(".") else text

    def fail(self, line: SourceLine, message: str) -> None:
        raise FlingusError(f"Line {line.number}: {message}\n  {line.text}")


DEMO_PROGRAM = """
BEGIN_PROGRAM "Flingus Demo"
  Store "Ada" in `name`.
  Store 3 in `count`.
  Store [] in `groceries`.

  Add "milk" to `groceries`.
  Add "bread" to `groceries`.
  Add "eggs" to `groceries`.

  Print "Hello, " followed by `name`.
  Print "Your list:".

  For each `item` in `groceries`:
    Print "- " followed by `item`.

  Count the items in `groceries` and store the result in `total`.

  If `total` is greater than 2:
    Print "You have several items to buy.".
  Otherwise:
    Print "Short list today.".

  While `count` is greater than 0:
    Print "Countdown: " followed by `count`.
    Calculate `count` minus 1 and store the result in `count`.

  Repeat 2 times:
    Print "Flingus is running.".
END_PROGRAM
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Flingus program.")
    parser.add_argument("file", nargs="?", help="A .flingus file to run.")
    parser.add_argument("--demo", action="store_true", help="Run the built-in demo program.")
    args = parser.parse_args(argv)

    if args.demo:
        code = DEMO_PROGRAM
    elif args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as source_file:
                code = source_file.read()
        except OSError as exc:
            print(f"Flingus could not open that file: {exc}", file=sys.stderr)
            return 1
    else:
        parser.print_help()
        return 2

    interpreter = FlingusInterpreter()
    try:
        interpreter.run(code)
    except FlingusError as exc:
        print("Flingus could not run your program.", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())