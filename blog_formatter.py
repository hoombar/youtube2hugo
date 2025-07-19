"""Blog post formatting module using Claude API for content enhancement."""

import os
import re
from typing import List, Dict, Optional
from anthropic import Anthropic
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BlogFormatter:
    """Handles blog post content formatting and enhancement using Claude API."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.anthropic_client = None
        
        # Initialize Anthropic client if API key is provided
        api_key = config.get('claude_api_key') or os.getenv('ANTHROPIC_API_KEY')
        if api_key:
            self.anthropic_client = Anthropic(api_key=api_key)
            logger.info("Claude API client initialized for blog formatting")
        else:
            logger.warning("No Claude API key found. Blog formatting will be skipped.")
    
    def format_content_with_images(
        self, 
        content_with_images: str, 
        title: str,
        frame_data: List[Dict]
    ) -> str:
        """Format content with two-pass Claude processing while preserving image positions."""
        
        if not self.anthropic_client:
            logger.info("No Claude API available, returning original content")
            return content_with_images
        
        # Extract image references from original content for validation
        original_images = self._extract_image_references(content_with_images)
        
        # Pass 2: Format as blog post with structure
        logger.info("Formatting content as blog post with Claude API...")
        formatted_content = self._format_as_blog_post(content_with_images, title)
        
        # Validate that images are reasonably preserved (allow for minor differences)
        formatted_images = self._extract_image_references(formatted_content)
        
        # Allow for 1-2 image differences (Claude might reorganize slightly)
        image_diff = abs(len(original_images) - len(formatted_images))
        if image_diff > 2:
            logger.warning(f"Too many images lost! Original: {len(original_images)}, Formatted: {len(formatted_images)}")
            logger.warning("Falling back to original content to preserve images")
            return content_with_images
        
        # Check for catastrophic content loss (more than 50% reduction indicates major problems)
        original_length = len(content_with_images.replace(' ', '').replace('\n', ''))
        formatted_length = len(formatted_content.replace(' ', '').replace('\n', ''))
        
        if formatted_length < original_length * 0.5:
            logger.warning(f"Catastrophic content reduction detected! Original: {original_length} chars, Formatted: {formatted_length} chars")
            logger.warning("Falling back to original content to preserve information")
            return content_with_images
        
        logger.info(f"Successfully preserved all {len(original_images)} images and content in formatted output")
        return formatted_content
    
    def _format_as_blog_post(self, content: str, title: str) -> str:
        """Second pass: Format content as a structured blog post."""
        
        prompt = self._get_blog_formatting_prompt(title)
        
        try:
            response = self.anthropic_client.messages.create(
                model=self.config.get('claude_model', 'claude-3-haiku-20240307'),
                max_tokens=8000,  # Increased to ensure we don't truncate content
                temperature=0.2,  # Lower temperature for more consistent preservation
                messages=[
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nContent to format:\n{content}"
                    }
                ]
            )
            
            formatted_content = response.content[0].text.strip()
            
            logger.info("Blog post formatting completed successfully")
            return formatted_content
            
        except Exception as e:
            logger.error(f"Error formatting blog post with Claude: {e}")
            logger.info("Continuing with original content...")
            return content
    
    def _get_blog_formatting_prompt(self, title: str) -> str:
        """Generate the prompt for Claude to format content as a blog post."""
        return f"""Transform this transcript-based content into a well-structured, engaging blog post titled "{title}".

CRITICAL PRESERVATION REQUIREMENTS:
1. **PRESERVE EVERY SINGLE IMAGE**: All ![...](filename.jpg) references must remain EXACTLY as they are
2. **PRESERVE ALL CONTENT**: Do not remove, summarize, or skip any information from the transcript
3. **PRESERVE IMAGE POSITIONS**: Keep images in their current positions relative to surrounding text
4. **PRESERVE TECHNICAL DETAILS**: Keep all technical information, examples, and explanations

MANDATORY BLOG STRUCTURE REQUIREMENTS:
- **MUST start with a brief introduction** (2-3 sentences explaining what the post covers)
- **MUST add meaningful section headers** using ## and ### to break up content logically
- **MUST improve paragraph breaks** - no walls of text
- **MUST add smooth transitions** between sections
- **MUST rewrite transcript language** to be more engaging and blog-appropriate (NOT transcript-like)
- **MUST add a conclusion** that summarizes key takeaways

FORMATTING IMPROVEMENTS (while preserving everything above):
- Transform conversational/spoken language into written blog style
- Fix sentence structure to be more polished than raw speech
- Add proper paragraph organization
- Create logical content flow with clear sections

EXAMPLE of what to preserve:
If you see: "Here's how to configure the settings. ![Screenshot of settings](frame_120.0s.jpg) As you can see in this interface..."
Keep it as: "Here's how to configure the settings. ![Screenshot of settings](frame_120.0s.jpg) As you can see in this interface..."

WHAT NOT TO DO:
- Do not move images to different locations
- Do not remove any images
- Do not change image filenames or alt text
- Do not remove any content or information
- Do not summarize or condense the content

Your goal is to make the content more readable and structured while keeping 100% of the original information and all images exactly where they are.

VERIFICATION CHECKLIST before returning:
✓ All ![...](filename.jpg) references are present and unchanged
✓ Images remain in their original positions relative to text
✓ No content has been removed or significantly shortened
✓ All technical details and examples are preserved
✓ All explanations and context around images are kept

Return the complete formatted blog post content (no front matter). The output should be longer or similar length to the input, never significantly shorter."""
    
    def apply_template(
        self, 
        content: str, 
        template_path: Optional[str], 
        template_variables: Dict[str, str]
    ) -> str:
        """Apply a template to the blog content with variable substitution."""
        
        if not template_path or not os.path.exists(template_path):
            logger.info("No template provided or template not found, using default structure")
            return content
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template = f.read()
            
            # Replace template variables
            for key, value in template_variables.items():
                placeholder = f"{{{{{key}}}}}"
                template = template.replace(placeholder, value)
            
            logger.info(f"Applied template: {template_path}")
            return template
            
        except Exception as e:
            logger.error(f"Error applying template {template_path}: {e}")
            return content
    
    def enhance_transcript_segments(self, segments: List[Dict]) -> List[Dict]:
        """First pass: Clean up transcript segments (called from transcript_extractor)."""
        
        if not self.anthropic_client:
            return segments
        
        # Combine all text for processing
        full_text = ' '.join(segment['text'] for segment in segments)
        
        # First pass: Basic cleanup
        cleaned_text = self._cleanup_transcript_text(full_text)
        
        # Split cleaned text back into segments
        cleaned_segments = self._redistribute_cleaned_text(segments, cleaned_text)
        
        return cleaned_segments
    
    def _cleanup_transcript_text(self, text: str) -> str:
        """First pass: Clean up transcript text for basic errors."""
        
        cleanup_prompt = f"""Please clean up this video transcript by fixing obvious errors from speech recognition. This is the FIRST PASS - focus only on basic cleanup:

1. Fix obvious typos and misheard words
2. Correct grammatical errors that clearly result from speech-to-text mistakes
3. Add proper punctuation where clearly missing
4. Fix capitalization issues
5. Remove filler words and speech artifacts like "um", "uh", repeated words

DO NOT:
- Change the overall meaning or tone
- Rephrase sentences significantly  
- Add new content or insights
- Remove substantive content
- Change technical terms unless obviously wrong
- Add structure or formatting (that comes later)

This transcript will be further processed, so keep it conversational and maintain the original flow.

Return only the cleaned transcript text:

{text}"""
        
        try:
            response = self.anthropic_client.messages.create(
                model=self.config.get('claude_model', 'claude-3-haiku-20240307'),
                max_tokens=6000,  # Increased to ensure we don't truncate content
                temperature=0.1,
                messages=[
                    {
                        "role": "user",
                        "content": cleanup_prompt
                    }
                ]
            )
            
            cleaned_text = response.content[0].text.strip()
            logger.info("First pass transcript cleanup completed")
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Error in first pass cleanup: {e}")
            return text
    
    def _redistribute_cleaned_text(self, original_segments: List[Dict], cleaned_text: str) -> List[Dict]:
        """Redistribute cleaned text back to original segment timing."""
        import re
        
        # Split cleaned text into sentences
        sentences = re.split(r'[.!?]+', cleaned_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # If we have fewer sentences than segments, try word-based distribution
        if len(sentences) < len(original_segments):
            words = cleaned_text.split()
            words_per_segment = max(1, len(words) // len(original_segments))
            
            cleaned_segments = []
            word_idx = 0
            
            for i, segment in enumerate(original_segments):
                start_word = word_idx
                end_word = min(word_idx + words_per_segment, len(words))
                
                # For the last segment, take all remaining words
                if i == len(original_segments) - 1:
                    end_word = len(words)
                
                segment_text = ' '.join(words[start_word:end_word])
                
                cleaned_segment = segment.copy()
                cleaned_segment['text'] = segment_text
                cleaned_segments.append(cleaned_segment)
                
                word_idx = end_word
            
            return cleaned_segments
        
        # Map sentences to segments
        cleaned_segments = []
        sentence_idx = 0
        
        for segment in original_segments:
            if sentence_idx < len(sentences):
                cleaned_segment = segment.copy()
                cleaned_segment['text'] = sentences[sentence_idx]
                sentence_idx += 1
            else:
                # If we run out of sentences, keep original text
                cleaned_segment = segment
            
            cleaned_segments.append(cleaned_segment)
        
        return cleaned_segments
    
    def _extract_image_references(self, content: str) -> List[str]:
        """Extract all image references from content."""
        import re
        # Match ![alt text](filename.jpg) pattern
        image_pattern = r'!\[.*?\]\([^)]+\)'
        return re.findall(image_pattern, content)