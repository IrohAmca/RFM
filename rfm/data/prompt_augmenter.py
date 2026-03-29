"""Prompt format augmentations to ensure format-invariant feature discovery.

By modifying the prompt format while keeping the core harmful/safe intent the same,
we can extract a wider variety of internal activations. This helps sparse autoencoders
(SAEs) learn features that are invariant to specific templates (e.g. conversational 
vs raw QA structure), preventing vulnerabilities to formatting-based adversarial attacks.
"""

from typing import List, Generator

class PromptAugmenter:
    """Diversify prompt formats for richer activation collection."""
    
    # 2025 finding: models process distinct intents differently if structured via 
    # Chat-markup vs QA formats vs raw completion. We supply 4 distinct contexts.
    TEMPLATES = [
        "{prompt}",                           # 1. Original format (no wrapper)
        "Please answer: {prompt}",            # 2. Polite/instructional wrapper
        "User: {prompt}\nAssistant:",         # 3. Chat/dialogue format
        "Question: {prompt}\nAnswer:",        # 4. Strict QA format
    ]
    
    def __init__(self, templates: List[str] = None):
        """Initialize with custom templates if provided, or defaults."""
        if templates is None:
            self.templates = self.TEMPLATES
        else:
            self.templates = templates
            
    def set_templates(self, templates: List[str]):
        """Override the active templates."""
        self.templates = templates
        
    def generate_variants(self, prompt: str) -> List[str]:
        """Generate all configured template variations for a given core prompt."""
        variants = []
        for tpl in self.templates:
            try:
                # Basic string formatting
                variant = tpl.format(prompt=prompt)
                variants.append(variant)
            except KeyError:
                # If template is invalid, fallback to raw prompt
                variants.append(prompt)
                
        # Deduplicate while preserving order mostly
        return list(dict.fromkeys(variants))

    def yield_augmented(self, raw_dataloader: Generator[dict, None, None], prompt_field: str = "prompt") -> Generator[dict, None, None]:
        """Wrap an existing dataloader generator to yield augmented variations instead of 1:1 items."""
        for row in raw_dataloader:
            base_prompt = row.get(prompt_field, "")
            if not base_prompt:
                yield row
                continue
                
            variants = self.generate_variants(base_prompt)
            for variant in variants:
                # Create a shallow copy of the row so metadata is duplicated
                aug_row = row.copy()
                aug_row[prompt_field] = variant
                aug_row["is_augmented"] = (variant != base_prompt)
                yield aug_row
