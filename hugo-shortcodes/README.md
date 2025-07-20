# Hugo Shortcodes for YouTube2Hugo Image Grids

If you're seeing "raw HTML omitted" in your Hugo blog posts, you have two options:

## Option 1: Enable HTML Rendering (Recommended)

Add this to your Hugo site's `config.yaml`:

```yaml
markup:
  goldmark:
    renderer:
      unsafe: true
```

Or if using `config.toml`:

```toml
[markup]
  [markup.goldmark]
    [markup.goldmark.renderer]
      unsafe = true
```

## Option 2: Use Hugo Shortcodes

1. Copy the shortcode files to your Hugo site:
   ```bash
   cp hugo-shortcodes/*.html your-hugo-site/layouts/shortcodes/
   ```

2. Update your YouTube2Hugo config:
   ```yaml
   hugo:
     use_shortcodes: true
   ```

3. Regenerate your blog posts.

## Shortcode Files

- **image-grid.html**: Container for multiple images in a grid layout
- **grid-image.html**: Individual image with optional timestamp caption

The shortcodes will generate the same grid layouts as the HTML version but using Hugo's native shortcode system instead of raw HTML.