from rfm.extractors.groq_generate import GroqGenerationExtractor


def test_groq_prompt_parser_preserves_system_and_user_roles():
    messages = GroqGenerationExtractor._messages_from_prompt(
        "[System]: Be concise.\n[User]: Hello\n[Assistant]:"
    )

    assert messages == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hello"},
    ]
