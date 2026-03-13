"""
Special token vocabulary builder for painting attributes.
Extracts and manages special tokens from captions.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict


class SpecialTokenVocabulary:
    """Manages special token vocabulary for painting attributes."""

    def __init__(self):
        self.token_categories = {
            'artist': set(),
            'year': set(),
            'gender': set(),
            'location': set()
        }
        self.token_to_id = {}
        self.id_to_token = {}
        self.category_to_tokens = defaultdict(list)

    def extract_tokens_from_jsonl(self, jsonl_path: str) -> Dict[str, Set[str]]:
        """Extract all special tokens from JSONL captions file."""
        token_pattern = re.compile(r'<([^>]+)>')

        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                caption = data['caption']

                # Find all special tokens
                tokens = token_pattern.findall(caption)

                for token in tokens:
                    full_token = f'<{token}>'

                    # Categorize token
                    if token.startswith('artist_'):
                        self.token_categories['artist'].add(full_token)
                    elif token.startswith('year_'):
                        self.token_categories['year'].add(full_token)
                    elif token.startswith('gender_'):
                        self.token_categories['gender'].add(full_token)
                    elif token.startswith('loc_'):
                        self.token_categories['location'].add(full_token)

        return self.token_categories

    def build_vocabulary(self) -> None:
        """Build token-to-id and id-to-token mappings."""
        idx = 0

        for category, tokens in self.token_categories.items():
            sorted_tokens = sorted(list(tokens))
            self.category_to_tokens[category] = sorted_tokens

            for token in sorted_tokens:
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token
                idx += 1

    def get_token_id(self, token: str) -> int:
        """Get ID for a token."""
        return self.token_to_id.get(token, -1)

    def get_token_from_id(self, token_id: int) -> str:
        """Get token string from ID."""
        return self.id_to_token.get(token_id, '<UNK>')

    def get_tokens_by_category(self, category: str) -> List[str]:
        """Get all tokens in a category."""
        return self.category_to_tokens.get(category, [])

    def get_token_category(self, token: str) -> str:
        """Determine which category a token belongs to."""
        if token.startswith('<artist_'):
            return 'artist'
        elif token.startswith('<year_'):
            return 'year'
        elif token.startswith('<gender_'):
            return 'gender'
        elif token.startswith('<loc_'):
            return 'location'
        return 'unknown'

    def extract_tokens_from_caption(self, caption: str) -> Dict[str, List[str]]:
        """Extract all special tokens from a single caption, organized by category."""
        token_pattern = re.compile(r'<([^>]+)>')
        tokens = token_pattern.findall(caption)

        categorized = {
            'artist': [],
            'year': [],
            'gender': [],
            'location': []
        }

        for token in tokens:
            full_token = f'<{token}>'
            category = self.get_token_category(full_token)
            if category != 'unknown':
                categorized[category].append(full_token)

        return categorized

    def remove_special_tokens(self, caption: str) -> str:
        """Remove all special tokens from caption, leaving only description."""
        token_pattern = re.compile(r'\s*<[^>]+>\s*')
        return token_pattern.sub(' ', caption).strip()

    def save_vocabulary(self, save_path: str) -> None:
        """Save vocabulary to JSON file."""
        vocab_data = {
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token,
            'category_to_tokens': self.category_to_tokens,
            'token_categories': {k: list(v) for k, v in self.token_categories.items()}
        }

        with open(save_path, 'w') as f:
            json.dump(vocab_data, f, indent=2)

    def load_vocabulary(self, load_path: str) -> None:
        """Load vocabulary from JSON file."""
        with open(load_path, 'r') as f:
            vocab_data = json.load(f)

        self.token_to_id = vocab_data['token_to_id']
        self.id_to_token = {int(k): v for k, v in vocab_data['id_to_token'].items()}
        self.category_to_tokens = vocab_data['category_to_tokens']
        self.token_categories = {
            k: set(v) for k, v in vocab_data['token_categories'].items()
        }

    def get_statistics(self) -> Dict:
        """Get vocabulary statistics."""
        return {
            'total_tokens': len(self.token_to_id),
            'num_artists': len(self.token_categories['artist']),
            'num_years': len(self.token_categories['year']),
            'num_genders': len(self.token_categories['gender']),
            'num_locations': len(self.token_categories['location'])
        }


def build_vocabulary_from_dataset(jsonl_path: str, save_path: str = None) -> SpecialTokenVocabulary:
    """
    Build special token vocabulary from dataset.

    Args:
        jsonl_path: Path to JSONL file with captions
        save_path: Optional path to save vocabulary

    Returns:
        SpecialTokenVocabulary object
    """
    vocab = SpecialTokenVocabulary()
    vocab.extract_tokens_from_jsonl(jsonl_path)
    vocab.build_vocabulary()

    if save_path:
        vocab.save_vocabulary(save_path)

    return vocab


if __name__ == '__main__':
    # Example usage
    dataset_path = '/home/wangyd/Projects/macs_thesis/yangyu/painting_content_tagged_1400_1600.jsonl'
    save_path = '/home/wangyd/Projects/macs_thesis/yangyu/special_token_vocab.json'

    vocab = build_vocabulary_from_dataset(dataset_path, save_path)

    print("Vocabulary Statistics:")
    print(json.dumps(vocab.get_statistics(), indent=2))

    print("\nExample tokens by category:")
    for category in ['artist', 'year', 'gender', 'location']:
        tokens = vocab.get_tokens_by_category(category)
        print(f"{category}: {tokens[:5]}... ({len(tokens)} total)")
