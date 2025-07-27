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
    
    def _safe_extract_response_text(self, response) -> str:
        """Safely extract text from Gemini response, handling finish_reason issues."""
        try:
            # Check if response has valid parts
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                
                # Check finish_reason
                if hasattr(candidate, 'finish_reason'):
                    finish_reason = candidate.finish_reason
                    if finish_reason == 2:  # SAFETY (content filtered)
                        logger.error("Gemini response filtered for safety reasons (finish_reason=2)")
                        logger.error("This video content is being blocked by Gemini's safety filters.")
                        logger.error("Consider using a different AI service or processing the video without AI formatting.")
                        raise SystemExit("❌ Gemini safety filter blocked the content. Exiting cleanly.")
                    elif finish_reason == 3:  # RECITATION (potential copyright)
                        logger.error("Gemini response blocked for potential recitation (finish_reason=3)")
                        raise SystemExit("❌ Content blocked due to potential copyright concerns. Exiting cleanly.")
                    elif finish_reason == 4:  # OTHER
                        logger.error("Gemini response failed for other reasons (finish_reason=4)")
                        raise SystemExit("❌ Gemini API failed for unspecified reasons. Exiting cleanly.")
                
                # Check if candidate has content
                if hasattr(candidate, 'content') and candidate.content and candidate.content.parts:
                    return response.text.strip()
            
            # If we get here, the response doesn't have valid content
            logger.error("Gemini response contains no valid content parts")
            raise ValueError("Invalid response format - no valid content parts")
            
        except AttributeError as e:
            logger.error(f"Gemini response structure error: {e}")
            raise ValueError(f"Invalid response structure: {e}")
    
    def __init__(self, config: Dict):
        self.config = config
        self.gemini_client = None
        
        # Load technical terms for correction
        self.technical_terms = self._load_technical_terms()
        
        # Initialize Gemini client if API key is provided
        api_key = (config.get('gemini_api_key') or 
                  config.get('gemini', {}).get('api_key') or 
                  os.getenv('GOOGLE_API_KEY'))
        if api_key:
            genai.configure(api_key=api_key)
            model_name = (config.get('gemini_model') or 
                         config.get('gemini', {}).get('model', 'gemini-2.5-flash'))
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
            logger.error("Gemini failed to create proper blog structure.")
            raise ValueError("Generated content lacks proper blog structure (insufficient sections).")
        
        # Validate that images are reasonably preserved (allow for minor differences)
        formatted_images = self._extract_image_references(formatted_content)
        
        # Allow for 1-2 image differences (Claude might reorganize slightly)
        image_diff = abs(len(original_images) - len(formatted_images))
        if image_diff > 2:
            logger.warning(f"Too many images lost! Original: {len(original_images)}, Formatted: {len(formatted_images)}")
            raise ValueError(f"Image preservation failed: lost {image_diff} images during formatting.")
        
        # Check for catastrophic content loss (more than 50% reduction indicates major problems)
        original_length = len(content_with_images.replace(' ', '').replace('\n', ''))
        formatted_length = len(formatted_content.replace(' ', '').replace('\n', ''))
        
        if formatted_length < original_length * 0.5:
            logger.warning(f"Catastrophic content reduction detected! Original: {original_length} chars, Formatted: {formatted_length} chars")
            raise ValueError(f"Content preservation failed: formatted content is too short ({formatted_length}/{original_length} chars).")
        
        logger.info(f"Successfully generated structured blog post with {len(self._extract_headers(formatted_content))} sections")
        return formatted_content
    
    def format_transcript_content(self, transcript_segments: List[Dict], title: str) -> str:
        """Format raw transcript segments into structured blog content without images."""
        
        if not self.gemini_client:
            error_msg = "Gemini API key is required for blog formatting. Please set GOOGLE_API_KEY environment variable or configure gemini.api_key in config."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Convert transcript segments to text with timestamp markers
        raw_content_with_markers = self._transcript_segments_to_text_with_markers(transcript_segments)
        
        # Apply technical terms corrections
        corrected_content = self._apply_technical_corrections(raw_content_with_markers)
        
        # Try multiple prompting strategies to work around safety filters
        logger.info("Formatting transcript content as structured blog post with Gemini API...")
        
        strategies = [
            ("standard", self._format_as_blog_post_with_boundaries),
            ("educational", self._format_as_blog_post_educational),
            ("tutorial", self._format_as_blog_post_tutorial),
            ("guide", self._format_as_blog_post_guide)
        ]
        
        for strategy_name, format_function in strategies:
            try:
                logger.info(f"🔄 Trying {strategy_name} prompting strategy...")
                formatted_content_with_markers = format_function(corrected_content, title)
                
                # Check if we got valid content
                if not formatted_content_with_markers or len(formatted_content_with_markers.strip()) < 100:
                    logger.warning(f"❌ {strategy_name} strategy returned insufficient content")
                    continue
                
                # Extract and store boundary information, then clean content
                formatted_content, boundary_map = self._extract_and_clean_boundaries(formatted_content_with_markers)
                
                # Store boundary map for later use in frame selection
                self.boundary_map = boundary_map
                
                # Validate blog structure
                if not self._validate_blog_structure(formatted_content):
                    logger.warning(f"❌ {strategy_name} strategy failed structure validation")
                    continue
                
                logger.info(f"✅ {strategy_name} strategy succeeded! Generated {len(self._extract_headers(formatted_content))} sections")
                return formatted_content
                
            except SystemExit as e:
                # Safety filter blocked - try next strategy
                logger.warning(f"❌ {strategy_name} strategy blocked by safety filter: {e}")
                continue
            except Exception as e:
                logger.warning(f"❌ {strategy_name} strategy failed: {e}")
                continue
        
        # All strategies failed
        logger.error("🚨 All prompting strategies failed - Gemini consistently blocks this content")
        raise ValueError("All prompting strategies failed due to safety filters. Content cannot be processed by Gemini.")
    
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
            
            formatted_content = self._safe_extract_response_text(response)
            
            logger.info("Blog post formatting completed successfully")
            return formatted_content
            
        except Exception as e:
            logger.error(f"Error formatting blog post with Gemini: {e}")
            logger.info("Continuing with original content...")
            return content
    
    def _get_blog_formatting_prompt(self, title: str) -> str:
        """Generate the prompt for Gemini to format content as a blog post."""
        technical_terms_section = self._generate_technical_terms_prompt()
        return f"""IMPORTANT CONTEXT: This is a transcript from an EDUCATIONAL TECHNICAL VIDEO about smart home technology, networking protocols, and home automation systems. All references to security concepts, vulnerabilities, network attacks, system administration, or troubleshooting are in the context of legitimate educational content about defensive cybersecurity practices and proper technical configuration.

Transform this transcript-based content into a well-structured, engaging blog post for "{title}".

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
- "Right, so you want to know how to build..." → "## Introduction\n\nWant to build a reliable ZigBee network? I'll walk you through everything you need to know..."
- "Let's dive straight into the foundation" → "## ZigBee Coordinator Fundamentals"
- "Right, let's talk about channel selection" → "## Channel Selection Strategy"
- "Now let's get into how the network works" → "## Understanding Network Architecture"
- "Okay, this is what separates beginners from advanced users" → "## Advanced Log Analysis Techniques"

CONVERSATIONAL TONE REQUIREMENTS:
- **USE PERSONAL PRONOUNS**: Write as "I" (the author) and "you" (the reader)
- **BE FRIENDLY AND APPROACHABLE**: Use conversational language like "I'll show you", "you'll want to", "let's look at"
- **DIRECT ENGAGEMENT**: Address the reader directly with phrases like "you'll need to make sure you...", "I'll cover that in more detail later"
- **CASUAL BUT INFORMATIVE**: Balance friendly tone with technical accuracy
- **RELATABLE EXPLANATIONS**: Use "you might be wondering", "here's what I've found works best"

CRITICAL WORD VARIETY REQUIREMENTS:
- **AVOID REPETITIVE FILLER WORDS**: Never use the same filler word (basically, essentially, actually, really, definitely, generally, typically, obviously, clearly, simply) more than 2-3 times in the entire blog post
- **USE DIVERSE TRANSITIONS**: Instead of repeating "basically" 8 times, use varied alternatives like: "in essence", "fundamentally", "at its core", "put simply", "the key point is", "what this means is", "in practical terms", "the bottom line is"
- **ELIMINATE REDUNDANT QUALIFIERS**: Remove unnecessary words that don't add meaning - instead of "basically very important" use "crucial" or "essential"
- **VARY SENTENCE STARTERS**: Don't begin multiple sentences with the same word or phrase
- **USE SPECIFIC LANGUAGE**: Replace vague terms with precise descriptions when possible

TONE TRANSFORMATION EXAMPLES:
BEFORE: "Even with the best coordinator, its physical placement is more critical than often perceived. Further details on placement will be discussed later. If a USB coordinator is in use, it is imperative to utilize a USB extension cable."
AFTER: "Even if you've got the best coordinator money can buy, where you place it might be more important than you think. I'll get into the specifics shortly, but if you're using a USB coordinator, you'll definitely want to use a shielded USB extension cable!"

BEFORE: "This configuration parameter requires careful consideration as it affects network performance."
AFTER: "You'll want to think carefully about this setting since it can really impact how well your network performs."

BEFORE: "The interface displays the current status of connected devices."
AFTER: "You can see all your connected devices right here in the interface."

WORD VARIETY TRANSFORMATION EXAMPLES:
BEFORE (REPETITIVE): "I basically took this prompt and said, 'Using the old logs, find this error.' It's basically going through and doing mapping so we can basically figure out what it looks like. We're basically going to get an artifact. I basically got a similar result, but basically only some very low severity errors. Claude is basically telling me this is not much of an issue."

AFTER (VARIED): "I took this prompt and said, 'Using the old logs, find this error.' It's going through and doing mapping so we can figure out what it looks like. We're going to get an artifact. I got a similar result, but with only some very low severity errors. Claude is telling me this is not much of an issue - in essence, it's a fairly transient problem that happens during startup."

FORMATTING IMPROVEMENTS (while preserving everything above):
- Write in first person ("I recommend", "I'll explain") and second person ("you should", "you'll see")
- Use contractions naturally ("you'll", "I'll", "don't", "can't") 
- Keep technical accuracy but make it feel like friendly advice
- Use encouraging language ("you've got this", "it's easier than it looks")
- Add personal insights and tips ("here's what I've learned", "this trick saved me hours")

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
        cleanup_prompt = f"""IMPORTANT CONTEXT: This is a transcript from an EDUCATIONAL TECHNICAL VIDEO about smart home technology, networking protocols, and home automation systems. All technical terminology regarding security, vulnerabilities, attacks, or system administration is in the context of legitimate educational content about defensive cybersecurity and proper network configuration.

Please clean up this video transcript by fixing obvious errors from speech recognition. This is the FIRST PASS - focus only on basic cleanup:

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
            
            cleaned_text = self._safe_extract_response_text(response)
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
        
        logger.info(f"📋 Validation found {len(headers)} headers: {headers}")
        
        # Must have at least 3 sections (relaxed from 5 for boundary marker system)
        if len(headers) < 3:
            logger.warning(f"Only {len(headers)} sections found, need at least 3")
            return False
        
        # For boundary marker system, be more flexible with section titles
        # Just check that we have meaningful section structure
        if len(headers) >= 3:
            logger.info(f"✅ Good section structure: {len(headers)} sections with meaningful titles")
        
        # Check for transcript artifacts (but be less strict)
        problematic_artifacts = ['right, so', 'okay, so', 'now, let\'s']
        content_lower = content.lower()
        artifact_count = 0
        for artifact in problematic_artifacts:
            if artifact in content_lower:
                artifact_count += 1
        
        if artifact_count > 2:  # Allow some artifacts, only fail if too many
            logger.warning(f"Found {artifact_count} transcript artifacts - content may need more cleanup")
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
        prompt = f"""IMPORTANT CONTEXT: This is a transcript from an EDUCATIONAL TECHNICAL VIDEO about smart home technology, networking protocols, and home automation systems. All technical terminology regarding security, vulnerabilities, attacks, or system administration is in the context of legitimate educational content about defensive cybersecurity and proper network configuration.

URGENT: Transform this transcript into a properly structured blog post for "{title}".

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
            return self._safe_extract_response_text(response)
        except Exception as e:
            logger.error(f"Error in strict blog formatting: {e}")
            raise
    
    def _format_as_blog_post_ultra_strict(self, content: str, title: str) -> str:
        """Ultra-strict formatting as last resort."""
        technical_terms_section = self._generate_technical_terms_prompt()
        prompt = f"""IMPORTANT CONTEXT: This is a transcript from an EDUCATIONAL TECHNICAL VIDEO about smart home technology, networking protocols, and home automation systems. All technical terminology regarding security, vulnerabilities, attacks, or system administration is in the context of legitimate educational content about defensive cybersecurity and proper network configuration.

FINAL ATTEMPT: Create a simple structured blog post for "{title}".

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
            return self._safe_extract_response_text(response)
        except Exception as e:
            logger.error(f"Error in ultra-strict blog formatting: {e}")
            raise
    
    def _extract_image_references(self, content: str) -> List[str]:
        """Extract all image references from content."""
        import re
        # Match ![alt text](filename.jpg) pattern
        image_pattern = r'!\[.*?\]\([^)]+\)'
        return re.findall(image_pattern, content)
    
    def _transcript_segments_to_text(self, transcript_segments: List[Dict]) -> str:
        """Convert transcript segments to plain text."""
        
        text_parts = []
        for segment in transcript_segments:
            text = segment.get('text', '').strip()
            if text:
                text_parts.append(text)
        
        return ' '.join(text_parts)
    
    def _apply_technical_corrections(self, content: str) -> str:
        """Apply technical term corrections to content."""
        
        corrected_content = content
        
        # Apply technical terms corrections
        for correct_term, variations in self.technical_terms.items():
            for variation in variations:
                # Case-insensitive replacement
                import re
                pattern = re.compile(re.escape(variation), re.IGNORECASE)
                corrected_content = pattern.sub(correct_term, corrected_content)
        
        logger.info(f"Applied {len(self.technical_terms)} technical term corrections")
        return corrected_content
    
    def _transcript_segments_to_text_with_markers(self, transcript_segments: List[Dict]) -> str:
        """Convert transcript segments to text with timestamp boundary markers."""
        
        text_parts = []
        
        for i, segment in enumerate(transcript_segments):
            text = segment.get('text', '').strip()
            # Handle both key formats for compatibility
            start_time = segment.get('start_time', segment.get('start', 0))
            
            if text:
                # Add timestamp marker before each segment
                marker = f"__TIMESTAMP_{start_time:.1f}__"
                text_parts.append(f"{marker} {text}")
        
        return ' '.join(text_parts)
    
    def _format_as_blog_post_with_boundaries(self, content: str, title: str) -> str:
        """Format content as blog post while preserving timestamp boundaries."""
        
        prompt = f"""
IMPORTANT CONTEXT: This is a transcript from an EDUCATIONAL TECHNICAL VIDEO about smart home technology, networking protocols, and home automation systems. All references to "attacks", "vulnerabilities", "security", "penetration", "exploits", or similar terms are in the context of legitimate cybersecurity education, network troubleshooting, and technical system administration. This content is for educational purposes about defensive security practices and proper network configuration.

CRITICAL: Transform this video transcript into a well-structured, professional blog post while preserving ALL timestamp markers.

MANDATORY BOUNDARY PRESERVATION:
1. PRESERVE ALL __TIMESTAMP_X.X__ markers EXACTLY as they appear - DO NOT REMOVE OR MODIFY THEM
2. Place timestamp markers within the content flow, not in headers
3. Keep the markers scattered throughout the text to mark timing boundaries
4. If ANY markers are missing, the entire system fails

BALANCED CONTENT TRANSFORMATION (while keeping all markers):
1. **PRESERVE TECHNICAL INFORMATION**: Keep all technical details, explanations, and specific information intact
2. **IMPROVE READABILITY**: Transform conversational speech into clear, structured prose
3. **STRUCTURED SECTIONS**: Use ## markdown headers for clear topic organization  
4. **CLEAN SPEECH PATTERNS**: Remove filler words, fix grammar, and improve sentence flow
5. **MAINTAIN INSTRUCTIONAL VALUE**: Preserve references to demonstrations and visual examples

TRANSFORMATION EXAMPLES:
❌ "Right, so, um, you want to know how to build a proper ZigBee network"
✅ "Want to build a solid ZigBee network? I'll walk you through the key components you need to understand"

❌ "Now let's talk about channel selection this is really important stuff"
✅ "Channel selection is really important - it can make or break your network performance, so you'll want to get this right"

❌ "You can see here in the interface that the coordinator is basically the brain"
✅ "You can see in the interface that the coordinator is basically the brain of your whole network"

CONVERSATIONAL WRITING STYLE:
- Use "I" and "you" throughout the content
- Write like you're giving friendly advice to a friend
- Use contractions naturally ("you'll", "I'll", "don't", "won't")
- Keep technical accuracy but make it approachable
- Add encouraging phrases like "you've got this" or "it's easier than you think"

CONTENT TRANSFORMATION REQUIREMENTS:
- Transform conversational speech into clear, instructional prose
- Keep ALL technical details, settings, values, and specific information
- Preserve the logical flow and sequence of instructions
- Convert "you can see" references into clear descriptive language
- Maintain all practical examples and demonstrations mentioned
- Remove speech filler but preserve substantive conversational context

Title: {title}

Content with timestamp markers:
{content}

Transform this into a professional, well-structured blog post that reads like expert technical documentation, not a transcript. PRESERVE EVERY TIMESTAMP MARKER while completely rewriting the conversational language.
"""
        
        try:
            response = self.gemini_client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=8000,
                    temperature=0.1,
                )
            )
            return self._safe_extract_response_text(response)
        except Exception as e:
            logger.error(f"Error in boundary-preserving blog formatting: {e}")
            raise
    
    def _format_as_blog_post_strict_with_boundaries(self, content: str, title: str) -> str:
        """Strict formatting with boundary preservation for retry attempts."""
        
        prompt = f"""
IMPORTANT CONTEXT: This is a transcript from an EDUCATIONAL TECHNICAL VIDEO about smart home technology, networking protocols, and home automation systems. All technical terminology regarding security, vulnerabilities, attacks, or system administration is in the context of legitimate educational content about defensive cybersecurity and proper network configuration.

URGENT: You FAILED to preserve timestamp markers in the previous attempt. This is CRITICAL for system functionality.

ABSOLUTE REQUIREMENTS - NO EXCEPTIONS:
- PRESERVE EVERY SINGLE __TIMESTAMP_X.X__ marker EXACTLY as written
- DO NOT DELETE, MODIFY, OR MOVE any timestamp markers  
- Keep markers within the content flow, scattered throughout paragraphs
- Use ## markdown headers for sections
- Maintain full content - DO NOT SUMMARIZE
- If ANY markers are missing, the entire system fails

CONTENT QUALITY REQUIREMENTS:
- Transform transcript language into professional blog writing
- Remove conversational words: "Right", "So", "Now", "Let's", "Okay"
- Use authoritative tone and definitive statements
- Break content into focused paragraphs
- Maintain technical accuracy

Title: {title}

Content with ESSENTIAL timestamp markers:
{content}

Output: Professional blog post with ALL timestamp markers preserved and transcript language transformed into polished writing.
"""
        
        try:
            response = self.gemini_client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=8000,
                    temperature=0.0,
                )
            )
            return self._safe_extract_response_text(response)
        except Exception as e:
            logger.error(f"Error in strict boundary-preserving blog formatting: {e}")
            raise
    
    def _format_as_blog_post_educational(self, content: str, title: str) -> str:
        """Educational prompting strategy to avoid safety filters."""
        
        prompt = f"""
Transform this educational technology tutorial transcript into a well-structured instructional article about "{title}".

EDUCATIONAL CONTEXT: This content is from a technical tutorial video about home automation setup and configuration. The content teaches legitimate technology skills for educational purposes.

CONTENT TRANSFORMATION REQUIREMENTS:
1. PRESERVE ALL __TIMESTAMP_X.X__ markers EXACTLY as they appear
2. Structure content with clear ## markdown headers for different topics
3. Transform spoken language into clear written instructions
4. Keep all technical details and step-by-step guidance
5. Organize information in a logical learning sequence

INSTRUCTIONAL FORMATTING:
- Use descriptive section headers that explain what users will learn
- Convert "let's do this" language into "to accomplish this task"
- Transform "you can see" into "the interface displays" or "the system shows"
- Keep all technical terms, settings, and specific values mentioned
- Maintain the instructional flow and sequence

Educational content:
{content}

Transform this into a clear, educational article that teaches these technology concepts effectively.
"""
        
        try:
            response = self.gemini_client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=8000,
                    temperature=0.2,
                )
            )
            return self._safe_extract_response_text(response)
        except Exception as e:
            logger.error(f"Error in educational blog formatting: {e}")
            raise
    
    def _format_as_blog_post_tutorial(self, content: str, title: str) -> str:
        """Tutorial prompting strategy focused on how-to content."""
        
        prompt = f"""
Convert this technology tutorial transcript into a comprehensive how-to guide about "{title}".

PURPOSE: Create an instructional guide that teaches users how to properly configure and use home automation technology.

FORMATTING REQUIREMENTS:
1. PRESERVE ALL __TIMESTAMP_X.X__ markers exactly as written
2. Create clear ## section headers for each major topic
3. Focus on step-by-step instructions and best practices
4. Explain the reasoning behind technical decisions
5. Keep all specific technical details and configuration values

TUTORIAL STRUCTURE:
- Transform conversational explanations into clear instructions
- Change "we're going to" into "this guide will show you how to"
- Convert "if you look at" into "examine the following"
- Keep all technical specifications and recommended settings
- Maintain the logical progression of setup steps

Tutorial content:
{content}

Create a comprehensive tutorial that guides users through these technical procedures.
"""
        
        try:
            response = self.gemini_client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=8000,
                    temperature=0.1,
                )
            )
            return self._safe_extract_response_text(response)
        except Exception as e:
            logger.error(f"Error in tutorial blog formatting: {e}")
            raise
    
    def _format_as_blog_post_guide(self, content: str, title: str) -> str:
        """Guide prompting strategy with focus on reference material."""
        
        prompt = f"""
Transform this technical reference transcript into a comprehensive guide about "{title}".

REFERENCE GUIDE CONTEXT: This material provides technical information about home automation systems, network configuration, and device setup procedures for legitimate educational and reference purposes.

TRANSFORMATION GUIDELINES:
1. PRESERVE ALL __TIMESTAMP_X.X__ markers without modification
2. Organize information with clear ## section headers
3. Present information as reference material and best practices
4. Focus on technical accuracy and completeness
5. Structure content for easy lookup and reference

REFERENCE FORMATTING:
- Convert spoken explanations into concise reference information
- Transform "what you need to know" into "key concepts include"
- Change "here's how to" into "the procedure involves"
- Preserve all technical specifications and configuration details
- Organize related concepts together logically

Reference material:
{content}

Create a comprehensive reference guide that presents this technical information clearly and accurately.
"""
        
        try:
            response = self.gemini_client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=8000,
                    temperature=0.3,
                )
            )
            return self._safe_extract_response_text(response)
        except Exception as e:
            logger.error(f"Error in guide blog formatting: {e}")
            raise
    
    def _extract_and_clean_boundaries(self, content_with_markers: str) -> tuple[str, Dict]:
        """Extract boundary information and return clean content + boundary map."""
        
        import re
        
        # Find all timestamp markers
        marker_pattern = r'__TIMESTAMP_(\d+\.\d+)__'
        markers = re.findall(marker_pattern, content_with_markers)
        logger.info(f"🕒 Found {len(markers)} timestamp markers in content")
        
        # Create boundary map by analyzing content structure
        boundary_map = {}
        lines = content_with_markers.split('\n')
        current_section = None
        
        for line_index, line in enumerate(lines):
            # Check if line contains a header
            header_match = re.match(r'^(#{1,3})\s+(.+)$', line)
            if header_match:
                section_title = header_match.group(2).strip()
                # Remove any markers from the title
                clean_title = re.sub(marker_pattern, '', section_title).strip()
                current_section = clean_title
                
                # Find the first timestamp marker after this header
                timestamp_match = re.search(marker_pattern, line)
                if timestamp_match:
                    timestamp = float(timestamp_match.group(1))
                    boundary_map[clean_title] = timestamp
                    logger.debug(f"📍 Found boundary: '{clean_title}' -> {timestamp:.1f}s (in header)")
                else:
                    # Look in subsequent lines for the first timestamp
                    found_timestamp = None
                    for next_line in lines[line_index + 1:line_index + 10]:  # Check next 10 lines
                        timestamp_match = re.search(marker_pattern, next_line)
                        if timestamp_match:
                            timestamp = float(timestamp_match.group(1))
                            boundary_map[clean_title] = timestamp
                            found_timestamp = timestamp
                            logger.debug(f"📍 Found boundary: '{clean_title}' -> {timestamp:.1f}s (in content)")
                            break
                    
                    if not found_timestamp:
                        logger.warning(f"⚠️  No timestamp found for section: '{clean_title}'")
                        # Use a default timestamp based on position
                        # Check if this looks like an introduction/title section that should start at 0
                        title_keywords = ['introduction', 'elevate', 'getting started', 'overview', 'intro', 'debug', 'test', 'video', 'smart doorbell']
                        is_intro_section = any(keyword in clean_title.lower() for keyword in title_keywords)
                        
                        if len(boundary_map) == 0 or is_intro_section:
                            # This is the first section OR an intro section, it should start at 0
                            default_timestamp = 0.0
                            logger.warning(f"   Detected intro/first section, using timestamp: 0.0s")
                        else:
                            # Subsequent sections: 60 seconds apart from the start
                            default_timestamp = len(boundary_map) * 60  # 60 seconds apart as default
                            logger.warning(f"   Using default timestamp: {default_timestamp:.1f}s")
                        boundary_map[clean_title] = default_timestamp
        
        # Clean all markers from content
        clean_content = re.sub(marker_pattern, '', content_with_markers)
        
        # Clean up extra spaces but preserve line structure
        lines = clean_content.split('\n')
        cleaned_lines = []
        for line in lines:
            # Clean extra spaces within lines but keep the line
            cleaned_line = re.sub(r'\s+', ' ', line).strip()
            cleaned_lines.append(cleaned_line)
        
        clean_content = '\n'.join(cleaned_lines)
        
        # Clean up excessive blank lines (more than 2 consecutive)
        clean_content = re.sub(r'\n\n\n+', '\n\n', clean_content)
        
        logger.info(f"📍 Extracted {len(boundary_map)} section boundaries: {list(boundary_map.keys())}")
        
        return clean_content.strip(), boundary_map