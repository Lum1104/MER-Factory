# MER-Factory GitHub Pages Documentation

Welcome to the GitHub Pages documentation site for MER-Factory!

## Local Development

To run this documentation site locally:

### Prerequisites

- Ruby 2.7+ (recommended: Ruby 3.0+)
- Bundler gem: `gem install bundler`

### Setup

```bash
cd docs/
bundle install
```

**Note**: This uses the `github-pages` gem which includes the exact Jekyll version used by GitHub Pages (currently Jekyll 3.9.x) to ensure local development matches the deployed site.

### Run Locally

```bash
# Start the Jekyll server
bundle exec jekyll serve

# Or with live reload
bundle exec jekyll serve --livereload

# Access at http://localhost:4000
```

### Troubleshooting

If you encounter dependency issues:

1. **Clean bundle cache**:
   ```bash
   bundle clean --force
   rm Gemfile.lock
   bundle install
   ```

2. **Ruby version issues**:
   ```bash
   # Check Ruby version
   ruby --version
   
   # If using rbenv, install compatible version
   rbenv install 3.1.0
   rbenv local 3.1.0
   ```

3. **GitHub Pages compatibility**:
   - The site uses `github-pages` gem for compatibility
   - This includes Jekyll 3.9.x (not 4.x) to match GitHub Pages
   - All plugins are compatible with GitHub Pages

## Structure

```
docs/
├── _config.yml          # Site configuration
├── _layouts/            # Custom layouts
│   └── default.html     # Main layout template
├── assets/             # Static assets
│   ├── css/
│   │   └── custom.css   # Custom styles
│   └── js/
│       └── custom.js    # Custom JavaScript
├── index.md            # Homepage
├── getting-started.md  # Getting started guide
├── api-reference.md    # API documentation
├── examples.md         # Examples and tutorials
├── technical-docs.md   # Technical documentation
└── Gemfile            # Ruby dependencies
```

## Theme

This site uses the **Jekyll Dinky theme** with extensive customizations:

- Modern responsive design
- Enhanced navigation
- Custom styling with CSS variables
- Interactive JavaScript features
- Mobile-optimized layout

## Content Guidelines

### Adding New Pages

1. Create a new `.md` file in the `docs/` directory
2. Add front matter with layout and metadata:
```yaml
---
layout: default
title: Page Title
description: Page description for SEO
---
```

3. Update navigation in `_config.yml` if needed

### Writing Documentation

- Use clear, descriptive headings
- Include code examples with syntax highlighting
- Add relevant links and cross-references
- Use appropriate Markdown formatting
- Include practical examples

### Style Guidelines

- Use **bold** for important terms
- Use `code` for technical terms and commands
- Use > blockquotes for important notes
- Include relevant emojis for visual appeal
- Keep paragraphs concise and readable

## Deployment

This site is automatically deployed to GitHub Pages when changes are pushed to the main branch. The documentation is available at:

https://Lum1104.github.io/MER-Factory/

## Customization

### Styling

Custom styles are in `assets/css/custom.css`. The design uses CSS custom properties for easy theming:

```css
:root {
  --primary-color: #2c3e50;
  --secondary-color: #3498db;
  --accent-color: #e74c3c;
  /* ... more variables */
}
```

### JavaScript

Interactive features are implemented in `assets/js/custom.js`:

- Scroll to top button
- Code copy functionality
- Table of contents generation
- Search functionality
- Keyboard navigation

### Layout

The main layout is in `_layouts/default.html` and includes:

- Responsive navigation
- Action buttons
- Footer with links
- SEO optimization
- Analytics support

## Contributing

To contribute to the documentation:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally with `bundle exec jekyll serve`
5. Submit a pull request

## Support

For questions about the documentation:

- 📚 Check existing documentation pages
- 🐛 Report issues on GitHub Issues
- 💬 Start a discussion on GitHub Discussions
- 📧 Contact the maintainers

---

Built with ❤️ using Jekyll and GitHub Pages
