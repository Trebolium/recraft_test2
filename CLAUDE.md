# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. That's all.

## Project context

This project is being developed under the constraints of a vibe-coding assessment. A rough proof-of-concept is preferable to a perfect system. The design will be continuously improved as the system is iterated over, so do not over-engineer or account for edge cases unless specifically asked to do so.

## Code style

For each code edit or file created, include more than enough comments and docstrings to make the code readable — but not so verbose that the documentation damages the flow of the code. The code needs to be intuitively readable at any stage of development.

## Environment

Install relevant Python packages and dependencies into the conda environment `recraft_test2`.

## Testing

When a user prompt specifies it, test adequately and consistently, and provide the command used to run the tests. Invoke tests by adding a `--test` flag to the command-line arguments when running the relevant Python script.

## Script structure

If a script grows too long or complicated, split it into multiple scripts, with a main script calling into the others.
