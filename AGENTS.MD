# Agents workflow

## 1. Always document
- Code should be documented with minimal but USEFUL comments, complicated functions or recursive functions should be commented further and all comments should follow a standard structure for rich comments.

- Every change resulting from each prompt should be documented inside the `changelog` folder. Each new document should be named `YYYYMMDD-HHMMSS-change.md` where the date and time are the time the change was made. The document should be formatted using markdown and should include the following information:
- Date and time
- Change summary
- Files modified
- Description of the changes
- JUSTIFICATION FOR THE CHANGES
- The prompt that caused it
Inside the changelog folder, there should be a `changelog.md` file that summarizes all the changes.

## 2. Code style
- Code should be written for the best performance and usability on ESP32 as well as interopacbility with other code and popular libraries
- It should follow a modern c++ standard
- It should be optimized
- It should have a reasoning behind what is the best solution, taking inspiration from already written solutions BEFORE WRITING NEW CODE

## 3. Workflow
- Before committing to a change, a plan detailing EVERY part of the changes that would need to be done from the prompt should be created and reviewed before proceeding with the changes, unless the user explicitly tells you to proceed without a plan.
- After implementing a solution from a prompt, tests should be created/updated to reflect the changes and ensure that the changes work as expected.
- When possible, visualizations should be created to demonstrate the changes and their effects.
- For Python `uv` should be used instead of pip and for `c++` pioarduino is used
- AI written implentations should live inside the src/impl/ai folder with a folder per-implementation
- When possible always try to import global modules from src/impl/main/globals.hpp (that's where most of the hardware interfaces are created)