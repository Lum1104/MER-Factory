# Contributing to MER-Factory

Thank you for your interest in contributing to MER-Factory! 🎉 This document provides guidelines and information for contributors to help you get started.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Development Guidelines](#development-guidelines)
- [Documentation](#documentation)
- [Community](#community)

## 🤝 Code of Conduct

This project follows a Code of Conduct to ensure a welcoming environment for all contributors. By participating, you agree to abide by these principles:

- **Be respectful**: Treat all community members with respect and kindness
- **Be inclusive**: Welcome contributors of all backgrounds and experience levels
- **Be constructive**: Provide helpful feedback and suggestions
- **Be patient**: Help others learn and grow in their contributions

## 🚀 Getting Started

### Prerequisites

Before contributing, ensure you have:

- **Python 3.12+** installed
- **Git** for version control
- **FFmpeg** installed and accessible in your PATH
- **OpenFace** compiled and configured (for AU/MER pipelines)
- Basic understanding of Python, async programming, and LLM concepts

### Areas for Contribution

We welcome contributions in several areas:

- 🐛 **Bug fixes** - Help resolve existing issues
- ✨ **New features** - Add new processing pipelines or analysis capabilities
- 📚 **Documentation** - Improve guides, examples, and API documentation
- 🧪 **Testing** - Expand test coverage and add integration tests
- 🎨 **UI/UX** - Enhance the dashboard interface and user experience
- 🔧 **Tooling** - Improve development workflow and build processes
- 🌐 **Model support** - Add support for new LLM providers or models

## 💻 Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/MER-Factory.git
cd MER-Factory

# Add the upstream repository
git remote add upstream https://github.com/Lum1104/MER-Factory.git
```

### 2. Environment Setup

```bash
# Create and activate virtual environment
conda create -n mer_factory python=3.12
conda activate mer_factory

# Install dependencies
pip install -r requirements.txt

# Copy environment file and configure
cp .env.example .env
# Edit .env with your API keys and OpenFace path
```

### 3. Verify Installation

```bash
# Test FFmpeg integration
python test/test_ffmpeg.py path/to/test_video.mp4 test_output/

# Test OpenFace integration (if configured)
python test/test_openface.py path/to/test_video.mp4 test_output/

# Run basic CLI test
python main.py --help
```

## 📝 Contributing Guidelines

### Workflow

1. **Create an issue** first to discuss your proposed changes
2. **Create a feature branch**
3. **Make your changes** following our coding standards
4. **Write or update tests** for your changes
5. **Update documentation** as needed
6. **Submit a pull request**

### Commit Messages

Use clear, descriptive commit messages:

```bash
# Good examples:
feat: add Qwen2-Audio model support for audio analysis
fix: resolve OpenFace path validation on Windows
docs: update installation guide with troubleshooting section
test: add integration tests for video processing pipeline

# Use conventional commit prefixes:
# feat: new feature
# fix: bug fix
# docs: documentation
# test: testing
# refactor: code refactoring
# style: formatting changes
# ci: CI/CD changes
```

## 🔄 Pull Request Process

### Before Submitting

- [ ] Code follows project style guidelines
- [ ] All tests pass locally
- [ ] Documentation is updated
- [ ] Changes are tested with different model providers
- [ ] PR description clearly explains the changes

### PR Template

When submitting a PR, use our [pull request template](.github/pull_request_template.md) and include:

- **Description**: What changes are made and why
- **Type of Change**: Bug fix, feature, documentation, etc.
- **Testing**: Which pipelines were verified
- **Screenshots**: For UI changes
- **Related Issues**: Link any related issues

### Architecture Principles

- **Modularity**: Keep components loosely coupled
- **Async-first**: Use async/await for I/O operations
- **Error handling**: Provide clear error messages and graceful degradation
- **Configuration**: Make features configurable through CLI or config files
- **Logging**: Use structured logging with appropriate levels

### Adding New Models

When adding support for new LLM providers:

1. **Create model class** in `mer_factory/models/api_models/` or `mer_factory/models/hf_models/`
2. **Implement required interfaces**
3. **Add configuration options** to `utils/config.py`
4. **Update model selection logic** in `mer_factory/models/__init__.py`
5. **Add tests** and documentation
6. **Update CLI help text** and examples

### Adding New Pipelines

For new processing pipelines:

1. **Define pipeline nodes** in `mer_factory/nodes/`
2. **Update state management** in `mer_factory/state.py`
3. **Modify graph structure** in `mer_factory/graph.py`
4. **Add CLI options** in `main.py`
5. **Create tests** and documentation

## 📖 Documentation

### Types of Documentation

- **Code documentation**: Docstrings and inline comments
- **User guides**: Setup, usage, and troubleshooting
- **API reference**: Function and class documentation
- **Examples**: Real-world usage scenarios
- **Technical docs**: Architecture and design decisions

### Documentation Standards

- Use **clear, concise language**
- Include **practical examples**
- Keep **screenshots up-to-date**
- Provide **troubleshooting sections**
- **Test all code examples**

### Building Documentation

```bash
# Serve documentation locally
cd docs/
bundle install
bundle exec jekyll serve
```

## 🌐 Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community discussions
- **Pull Requests**: Code review and technical discussions

### Getting Help

- **Check existing issues** before creating new ones
- **Use descriptive titles** for issues and discussions
- **Provide context** and examples when asking questions
- **Be patient and respectful** in all interactions

### Recognition

Contributors are recognized through:

- **GitHub contributor stats**
- **Release notes** mentioning significant contributions
- **Documentation credits**
- **Community acknowledgments**

## 🎯 Project Roadmap

Check our [project roadmap](https://github.com/Lum1104/MER-Factory/wiki) for:

- Planned features and improvements
- Current development priorities
- Long-term project goals
- Opportunities for contribution

## 📄 License

By contributing to MER-Factory, you agree that your contributions will be licensed under the [MIT License](LICENSE).

---

**Thank you for contributing to MER-Factory!** 🙏

Your contributions help advance multimodal emotion recognition research and make this tool better for the entire community. If you have questions about contributing, feel free to open an issue or start a discussion.

For additional resources:

- 📖 **[Documentation](https://lum1104.github.io/MER-Factory/)**
- 🔧 **[Technical Docs](https://lum1104.github.io/MER-Factory/technical-docs)**
- 💡 **[Examples](https://lum1104.github.io/MER-Factory/examples)**
- 🛠️ **[Tools & Dashboard](https://lum1104.github.io/MER-Factory/tools)**
