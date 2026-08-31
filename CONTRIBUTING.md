# Contributing Guidelines

Thank you for considering contributing to our project! Please follow the guidelines below to ensure consistency and quality.

---

## Development setup

`pyproject.toml` is the source of truth for dependencies. From a clone of your fork:

```bash
pip install -e ".[dev,plotting]"
```

The `dev` extra includes pytest, Black, and Flake8. The `plotting` extra (matplotlib) is needed for the visualization tests.

## Code Style & Quality

- **Formatting:** Use **Black** (`==26.5.1`) for code formatting.
- **Linting:** Use **Flake8** for code quality checks.
- **Typing:** Add type hints for function parameters and return values.
- **Structure:**  
  - Follow the project directory structure.  
  - Keep functions small and focused (single responsibility).  
- **Error Handling:**  
  - Implement proper exception handling.  
  - Validate inputs and provide descriptive error messages.  
- **Testing:**  
  - Write unit tests (unittest-style `TestCase` is fine) and place them in `tests/` mirroring the package structure.  
  - Run tests locally with pytest before submitting a PR:

  ```bash
  pytest
  ```

## Pull Request Process

1. Fork the repository and create a feature branch (`git checkout -b feature/amazing-feature`)
2. Make your changes and ensure formatting and linting pass:

   ```bash
   black CausalEstimate tests
   flake8 CausalEstimate tests --select=E9,F63,F7,F82,U100,E711,E712,E713,E714,E721,F401,F402,F405,F811,F821,F822,F823,F831,F841,F901,
   ```

3. Commit using conventional commit messages:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation
   - `test:` for adding tests
   - `refactor:` for code refactoring

4. Push to your fork and open a Pull Request targeting `main`. CI (pytest, Black, Flake8) runs on pull requests.

### PR Requirements

- Follow existing code style
- Add tests for new features
- Update documentation as needed
- All CI checks must pass
- Keep changes focused and atomic

Thank you for contributing! 🚀
