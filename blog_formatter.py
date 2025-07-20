"""Blog post formatting module using Gemini API for content enhancement."""

import os
import re
from typing import List, Dict, Optional
import google.generativeai as genai
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BlogFormatter:
    """Handles blog post content formatting and enhancement using Gemini API."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.gemini_client = None
        
        # Load technical terms for correction
        self.technical_terms = self._load_technical_terms()
        
        # Initialize Gemini client if API key is provided
        api_key = config.get('gemini_api_key') or os.getenv('GOOGLE_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            model_name = config.get('gemini_model', 'gemini-2.5-flash')
            self.gemini_client = genai.GenerativeModel(model_name)
            logger.info(f"Gemini API client initialized for blog formatting with model: {model_name}")
        else:
            logger.warning("No Gemini API key found. Blog formatting will be skipped.")
    
    def format_content_with_images(
        self, 
        content_with_images: str, 
        title: str,
        frame_data: List[Dict]
    ) -> str:
        """Format content with two-pass Claude processing while preserving image positions."""
        
        if not self.gemini_client:
            error_msg = "Gemini API key is required for blog formatting. Please set GOOGLE_API_KEY environment variable or configure gemini.api_key in config."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Extract image references from original content for validation
        original_images = self._extract_image_references(content_with_images)
        
        # Enhance image context before formatting
        enhanced_content = self._enhance_image_context(content_with_images, frame_data)
        
        # Pass 2: Format as blog post with structure
        logger.info("Formatting content as blog post with Gemini API...")
        formatted_content = self._format_as_blog_post(enhanced_content, title)
        
        # Validate that the output has proper blog structure
        if not self._validate_blog_structure(formatted_content):
            logger.error("Gemini failed to create proper blog structure. Retrying...")
            # Try once more with stronger prompt
            formatted_content = self._format_as_blog_post_strict(enhanced_content, title)
            if not self._validate_blog_structure(formatted_content):
                logger.error("Gemini failed to format content properly after retry")
                formatted_content = self._handle_gemini_failure_interactive(enhanced_content, title, "blog structure validation")
                if formatted_content is None:
                    raise ValueError("Failed to generate properly structured blog post. User chose to exit.")
        
        # Validate that images are reasonably preserved (allow for minor differences)
        formatted_images = self._extract_image_references(formatted_content)
        
        # Allow for 1-2 image differences (Claude might reorganize slightly)
        image_diff = abs(len(original_images) - len(formatted_images))
        if image_diff > 2:
            logger.warning(f"Too many images lost! Original: {len(original_images)}, Formatted: {len(formatted_images)}")
            retry_content = self._handle_gemini_failure_interactive(enhanced_content, title, "image preservation")
            if retry_content is None:
                raise ValueError("Image preservation failed during formatting. User chose to exit.")
            formatted_content = retry_content
        
        # Check for catastrophic content loss (more than 50% reduction indicates major problems)
        original_length = len(content_with_images.replace(' ', '').replace('\n', ''))
        formatted_length = len(formatted_content.replace(' ', '').replace('\n', ''))
        
        if formatted_length < original_length * 0.5:
            logger.warning(f"Catastrophic content reduction detected! Original: {original_length} chars, Formatted: {formatted_length} chars")
            retry_content = self._handle_gemini_failure_interactive(enhanced_content, title, "content preservation")
            if retry_content is None:
                raise ValueError("Content preservation failed during formatting. User chose to exit.")
            formatted_content = retry_content
        
        logger.info(f"Successfully generated structured blog post with {len(self._extract_headers(formatted_content))} sections")
        return formatted_content
    
    def _handle_gemini_failure_interactive(self, content: str, title: str, failure_type: str) -> str:
        """Handle Gemini API failures with interactive retry options."""
        max_retries = self.config.get('gemini', {}).get('max_retries', 3)
        retry_count = 0
        
        print(f"\n❌ Gemini API failed during {failure_type}")
        print("This could be due to:")
        print("  - Temporary API issues")
        print("  - Rate limiting")
        print("  - Content complexity")
        print("  - Network connectivity issues")
        
        while retry_count < max_retries:
            print(f"\n🔄 Retry attempt {retry_count + 1}/{max_retries}")
            print("Options:")
            print("  [Enter] - Retry with Gemini API")
            print("  [s] - Skip Gemini formatting (use raw content)")
            print("  [q] - Quit and exit script")
            
            try:
                choice = input("Your choice: ").lower().strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\n👋 User interrupted. Exiting...")
                return None
            
            if choice == 'q':
                print("👋 User chose to exit.")
                return None
            elif choice == 's':
                print("⚠️  Skipping Gemini formatting. Using raw content.")
                logger.warning("User chose to skip Gemini formatting")
                return content  # Return unformatted content
            else:
                # Default: retry
                print("🔄 Retrying with Gemini API...")
                try:
                    if failure_type == "blog structure validation":
                        # Try different formatting approach
                        if retry_count == 0:
                            result = self._format_as_blog_post_strict(content, title)
                        else:
                            # Try with even more explicit prompting
                            result = self._format_as_blog_post_ultra_strict(content, title)
                    else:
                        # For image/content preservation issues, try standard formatting
                        result = self._format_as_blog_post(content, title)
                    
                    # Re-validate the result
                    if failure_type == "blog structure validation":
                        if self._validate_blog_structure(result):
                            print("✅ Retry successful!")
                            return result
                        else:
                            print("❌ Retry failed - structure still invalid")
                    elif failure_type == "image preservation":
                        formatted_images = self._extract_image_references(result)
                        original_images = self._extract_image_references(content)
                        image_diff = abs(len(original_images) - len(formatted_images))
                        if image_diff <= 2:
                            print("✅ Retry successful!")
                            return result
                        else:
                            print(f"❌ Retry failed - still losing {image_diff} images")
                    elif failure_type == "content preservation":
                        original_length = len(content.replace(' ', '').replace('\n', ''))
                        formatted_length = len(result.replace(' ', '').replace('\n', ''))
                        if formatted_length >= original_length * 0.5:
                            print("✅ Retry successful!")
                            return result
                        else:
                            print(f"❌ Retry failed - content still too short ({formatted_length}/{original_length} chars)")
                    
                except Exception as e:
                    print(f"❌ Retry failed with error: {e}")
                
                retry_count += 1
        
        print(f"\n❌ All {max_retries} retries failed.")
        print("Final options:")
        print("  [s] - Skip Gemini formatting (use raw content)")
        print("  [q] - Quit and exit script")
        
        try:
            final_choice = input("Your choice: ").lower().strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 User interrupted. Exiting...")
            return None
        
        if final_choice == 's':
            print("⚠️  Using raw content without Gemini formatting.")
            logger.warning("User chose to skip Gemini formatting after all retries failed")
            return content
        else:
            print("👋 User chose to exit after retries failed.")
            return None
    
    def _format_as_blog_post(self, content: str, title: str) -> str:
        """Second pass: Format content as a structured blog post."""
        
        prompt = self._get_blog_formatting_prompt(title)
        
        try:
            full_prompt = f"{prompt}\n\nContent to format:\n{content}"
            
            response = self.gemini_client.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=8000,
                    temperature=0.2,  # Lower temperature for more consistent preservation
                )
            )
            
            formatted_content = response.text.strip()
            
            logger.info("Blog post formatting completed successfully")
            return formatted_content
            
        except Exception as e:
            logger.error(f"Error formatting blog post with Gemini: {e}")
            logger.info("Continuing with original content...")
            return content
    
    def _get_blog_formatting_prompt(self, title: str) -> str:
        """Generate the prompt for Gemini to format content as a blog post."""
        technical_terms_section = self._generate_technical_terms_prompt()
        return f"""Transform this transcript-based content into a well-structured, engaging blog post for "{title}".

IMPORTANT: DO NOT include the title "{title}" anywhere in your output - it will be added separately in the template.

{technical_terms_section}

Pay special attention to company names, product names, and technical protocols that are commonly mispronounced or misheard in speech-to-text.

CRITICAL PRESERVATION REQUIREMENTS:
1. **PRESERVE EVERY SINGLE IMAGE**: All ![...](filename.jpg) references must remain EXACTLY as they are
2. **PRESERVE ALL CONTENT**: Do not remove, summarize, or skip any information from the transcript
3. **PRESERVE IMAGE POSITIONS**: Keep images in their current positions relative to surrounding text
4. **PRESERVE TECHNICAL DETAILS**: Keep all technical information, examples, and explanations

MANDATORY BLOG STRUCTURE REQUIREMENTS:
- **MUST start with a compelling introduction** using "## Introduction" header (NOT the title)
- **MUST create AT LEAST 5-7 clear section headers** using ## format for main topics
- **MUST break up ALL long paragraphs** - maximum 3-4 sentences per paragraph
- **MUST eliminate ALL transcript language** (remove "Right", "So", "Now", "Let's", "Okay")
- **MUST add a "## Conclusion" section** that summarizes key takeaways

CRITICAL SECTION CREATION RULES:
- EVERY major topic change MUST have a ## header
- Look for topic transitions like equipment discussion → placement → channel selection → troubleshooting
- Convert ANY mention of "let's talk about X" into "## X"
- Convert ANY mention of "now we're going to cover Y" into "## Y" 
- Add ## headers before discussing ANY new concept or tool

REQUIRED SECTION STRUCTURE (YOU MUST USE THESE):
## Introduction
## [Topic 1 - extract from content]
## [Topic 2 - extract from content] 
## [Topic 3 - extract from content]
## [Topic 4 - extract from content]
## [Topic 5 - extract from content]
## Conclusion

AGGRESSIVE TRANSFORMATION EXAMPLES:
- "Right, so you want to know how to build..." → "## Introduction\n\nBuilding a reliable ZigBee network..."
- "Let's dive straight into the foundation" → "## ZigBee Coordinator Fundamentals"
- "Right, let's talk about channel selection" → "## Channel Selection Strategy"
- "Now let's get into how the network works" → "## Understanding Network Architecture"
- "Okay, this is what separates beginners from advanced users" → "## Advanced Log Analysis Techniques"

FORMATTING IMPROVEMENTS (while preserving everything above):
- Transform conversational/spoken language into polished written style
- Remove transcript artifacts like "you can see here", "as the video goes on"
- Fix sentence structure to be more engaging than raw speech
- Add proper paragraph organization with clear topic sentences
- Create logical content flow with seamless transitions

EXAMPLE TRANSFORMATION:
BEFORE: "Right, so you want to know how to build a proper ZigBee network that actually works reliably, or maybe you're doing battle with your current existing mesh network."
AFTER: "## Introduction\n\nBuilding a reliable ZigBee network can be challenging, whether you're starting from scratch or troubleshooting an existing mesh setup."

WHAT NOT TO DO:
- Do not move images to different locations
- Do not remove any images or change their filenames
- Do not remove any technical content or information
- Do not summarize or condense explanations
- Do not change the meaning or lose any details

Your goal is to transform transcript-style content into engaging blog format while preserving every detail and image.

CRITICAL VERIFICATION CHECKLIST - OUTPUT WILL BE REJECTED IF ANY ITEM IS MISSING:
✓ Output contains AT LEAST 5 section headers starting with "##"
✓ First section is "## Introduction"
✓ Last section is "## Conclusion" 
✓ NO paragraphs longer than 4 sentences
✓ NO transcript language ("Right", "So", "Now", "Let's", "Okay")
✓ All ![...](filename.jpg) references are present and unchanged
✓ Images remain in their original positions relative to text
✓ All technical content and examples are preserved

EXAMPLE OF WHAT YOUR OUTPUT SHOULD LOOK LIKE:
## Introduction
[2-3 engaging sentences about the topic]

## [Clear Topic Header]
[3-4 sentence paragraph about this topic]

![Alt text](image.jpg)

[Another 3-4 sentence paragraph]

## [Another Clear Topic Header]
[Content continues with proper structure]

## Conclusion
[Summary and key takeaways]

Return the complete formatted blog post content (no front matter). Must contain clear ## section headers throughout."""
    
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
        
        if not self.gemini_client:
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
        
        technical_terms_section = self._generate_technical_terms_prompt()
        cleanup_prompt = f"""Please clean up this video transcript by fixing obvious errors from speech recognition. This is the FIRST PASS - focus only on basic cleanup:

1. Fix obvious typos and misheard words
2. Correct grammatical errors that clearly result from speech-to-text mistakes
3. Add proper punctuation where clearly missing
4. Fix capitalization issues
5. Remove filler words and speech artifacts like "um", "uh", repeated words

{technical_terms_section}

DO NOT:
- Change the overall meaning or tone
- Rephrase sentences significantly  
- Add new content or insights
- Remove substantive content
- Change technical terms unless obviously wrong (use the correction list above)
- Add structure or formatting (that comes later)

This transcript will be further processed, so keep it conversational and maintain the original flow.

Return only the cleaned transcript text:

{text}"""
        
        try:
            response = self.gemini_client.generate_content(
                cleanup_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=6000,
                    temperature=0.1,
                )
            )
            
            cleaned_text = response.text.strip()
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
    
    def _enhance_image_context(self, content: str, frame_data: List[Dict]) -> str:
        """Enhance image alt text and context based on frame data."""
        import re
        
        # Create a mapping of timestamps to frame descriptions
        frame_map = {}
        for frame in frame_data:
            timestamp = frame.get('timestamp', 0)
            description = frame.get('description', '')
            frame_map[timestamp] = description
        
        # Find and enhance image references
        def enhance_image_reference(match):
            full_match = match.group(0)
            # Extract timestamp from filename (e.g., frame_15.0s_optimized.jpg -> 15.0)
            timestamp_match = re.search(r'frame_(\d+(?:\.\d+)?)s', full_match)
            if timestamp_match:
                timestamp = float(timestamp_match.group(1))
                if timestamp in frame_map and frame_map[timestamp]:
                    # Use the AI-generated description as alt text
                    description = frame_map[timestamp]
                    # Update alt text in the image reference
                    enhanced = re.sub(r'!\[.*?\]', f'![{description}]', full_match)
                    return enhanced
            return full_match
        
        # Apply enhancements to all image references
        image_pattern = r'!\[.*?\]\([^)]+\)'
        enhanced_content = re.sub(image_pattern, enhance_image_reference, content)
        
        return enhanced_content
    
    def _load_technical_terms(self) -> Dict[str, str]:
        """Load technical terms from configuration with default Home Assistant terms."""
        # Default Home Assistant and smart home terms
        default_terms = {
            'Home Assistant': ['home assistant', 'homeassistant', 'Home-Assistant', 'home-assistant'],
            'Nabu Casa': ['Nabakaza', 'Naba Casa', 'Nava Casa', 'Nabu-Casa', 'nabucasa'],
            'ZigBee': ['Zigby', 'Zigbee', 'Zig Bee', 'Zig-Bee', 'zigby'],
            'Z-Wave': ['Z Wave', 'Zwave', 'Z-way', 'z-wave', 'zwave'],
            'Node-RED': ['Node Red', 'NodeRed', 'Node-red', 'node-red', 'nodred'],
            'MQTT': ['MQ TT', 'EMQTT', 'M QTT', 'mqtt', 'Mqtt'],
            'ESPHome': ['ESP Home', 'ESP-Home', 'Esp Home', 'esp-home', 'esphome'],
            'Frigate': ['Friggit', 'Friget', 'Frigit', 'frigate'],
            'Hass.io': ['Hassio', 'Has.io', 'Hass io', 'hassio', 'hass-io'],
            'Add-on': ['Addon', 'Add on', 'ad-on', 'addon'],
            'Supervisor': ['supervisor'],
            'HACS': ['Hax', 'HACs', 'H-A-C-S', 'hacs'],
            'Lovelace': ['Love Lace', 'Lovelase', 'Love-lace', 'lovelace'],
            'Zigbee2MQTT': ['Zigbee to MQTT', 'Zigbee 2 MQTT', 'Zigby2MQTT', 'zigbee2mqtt'],
            'ConBee': ['Con Bee', 'Conby', 'ConBy', 'con-bee', 'conbee'],
            'deCONZ': ['de-CONZ', 'deconz', 'De Conz', 'de-conz'],
            'InfluxDB': ['Influx DB', 'influxdb', 'Influx-DB', 'influx-db'],
            'Grafana': ['Graphana', 'Grafanna', 'Graf-ana', 'grafana'],
            'Prometheus': ['Promethius', 'Prometheous', 'prometheus'],
            'Docker': ['docker'],
            'Portainer': ['Port-ainer', 'Portaner', 'Port Ainer', 'portainer']
        }
        
        # Load additional terms from config if available
        config_terms = self.config.get('technical_terms', {})
        
        # Merge config terms with defaults
        all_terms = default_terms.copy()
        all_terms.update(config_terms)
        
        return all_terms
    
    def _generate_technical_terms_prompt(self) -> str:
        """Generate the technical terms correction section for prompts."""
        terms_text = "CRITICAL TECHNICAL TERM CORRECTIONS:\nPay special attention to correcting these commonly misheard technical terms:\n"
        
        for correct_term, incorrect_variants in self.technical_terms.items():
            variants_str = ', '.join([f'"{variant}"' for variant in incorrect_variants])
            terms_text += f"- {correct_term} (not {variants_str})\n"
        
        return terms_text
    
    def _validate_blog_structure(self, content: str) -> bool:
        """Validate that content has proper blog structure with sections."""
        headers = self._extract_headers(content)
        
        # Must have at least 5 sections
        if len(headers) < 5:
            logger.warning(f"Only {len(headers)} sections found, need at least 5")
            return False
        
        # Must start with Introduction
        if not headers[0].lower().strip().startswith('introduction'):
            logger.warning(f"First section is '{headers[0]}', should be 'Introduction'")
            return False
        
        # Must end with Conclusion
        if not headers[-1].lower().strip().startswith('conclusion'):
            logger.warning(f"Last section is '{headers[-1]}', should be 'Conclusion'")
            return False
        
        # Check for transcript artifacts
        transcript_artifacts = ['right,', 'so,', "let's", 'now,', 'okay,']
        content_lower = content.lower()
        for artifact in transcript_artifacts:
            if artifact in content_lower:
                logger.warning(f"Found transcript artifact: '{artifact}'")
                return False
        
        return True
    
    def _extract_headers(self, content: str) -> List[str]:
        """Extract all ## headers from content."""
        import re
        header_pattern = r'^## (.+)$'
        headers = re.findall(header_pattern, content, re.MULTILINE)
        return headers
    
    def _format_as_blog_post_strict(self, content: str, title: str) -> str:
        """Strict formatting with even stronger requirements."""
        technical_terms_section = self._generate_technical_terms_prompt()
        prompt = f"""URGENT: Transform this transcript into a properly structured blog post for "{title}".

IMPORTANT: DO NOT include the title "{title}" anywhere in your output - it will be added separately.

{technical_terms_section}

THIS IS YOUR SECOND ATTEMPT. THE FIRST ATTEMPT FAILED VALIDATION.

MANDATORY REQUIREMENTS (YOUR OUTPUT WILL BE REJECTED IF MISSING):
1. Must start with exactly "## Introduction" (NOT the title)
2. Must have AT LEAST 5 total sections with ## headers
3. Must end with exactly "## Conclusion"  
4. NO transcript words: "Right", "So", "Let's", "Now", "Okay" anywhere
5. Maximum 3 sentences per paragraph
6. All images ![...](file.jpg) must be preserved exactly

EXAMPLE STRUCTURE YOU MUST FOLLOW:
## Introduction
Brief engaging intro about the topic.

## [Topic Name]
Short paragraph about this topic. Never more than 3 sentences.

![Description](image.jpg)

Another short paragraph. Keep it concise.

## [Next Topic]
Continue with proper structure.

## Conclusion
Summary of key points.

Content to transform:
{content}

CRITICAL: Your output MUST start with "## Introduction" and end with "## Conclusion"."""
        
        try:
            response = self.gemini_client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=8000,
                    temperature=0.1,  # Very low temperature for consistent structure
                )
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error in strict blog formatting: {e}")
            raise
    
    def _format_as_blog_post_ultra_strict(self, content: str, title: str) -> str:
        """Ultra-strict formatting as last resort."""
        technical_terms_section = self._generate_technical_terms_prompt()
        prompt = f"""FINAL ATTEMPT: Create a simple structured blog post for "{title}".

IMPORTANT: DO NOT include the title "{title}" anywhere in your output.

{technical_terms_section}

SIMPLE REQUIREMENTS:
1. Start with: ## Introduction
2. Add 3-4 sections with ## headers
3. End with: ## Conclusion
4. Preserve ALL images ![...](filename) exactly
5. Use simple, clear language

MINIMAL STRUCTURE:
## Introduction
[Brief intro paragraph]

## Main Content
[Key content from transcript]

![image](filename.jpg)

[More content]

## Key Points
[Important points]

## Conclusion
[Brief conclusion]

Transform this content:
{content}

Output a simple, well-structured blog post following the exact format above."""
        
        try:
            response = self.gemini_client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=6000,
                    temperature=0.0,  # Zero temperature for maximum consistency
                )
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error in ultra-strict blog formatting: {e}")
            raise
    
    def _extract_image_references(self, content: str) -> List[str]:
        """Extract all image references from content."""
        import re
        # Match ![alt text](filename.jpg) pattern
        image_pattern = r'!\[.*?\]\([^)]+\)'
        return re.findall(image_pattern, content)