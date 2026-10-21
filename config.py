from guardrails import Guard
from guardrails_ai.regex_match import RegexMatch
# Import Guard and Validator
from guardrails_ai.gibberish_text import GibberishText
# Import Guard and Validator
# from guardrails_ai.politeness_check import PolitenessCheck
from guardrails_ai.saliency_check import SaliencyCheck
# Import Guard and Validator
from guardrails_ai.guardrails_pii import GuardrailsPII
from guardrails_ai.detect_prompt_injection import DetectPromptInjection

# create a pinecone index called "detect-prompt-injection" before running this
guard_prompt = Guard().use(
    DetectPromptInjection(
        pinecone_index="detect-prompt-injection",
        on_fail="exception",
    )
)
# Setup Guard
guard_PII = Guard().use(
    GuardrailsPII(entities=["DATE_TIME"], on_fail="fix")
)

# Initialize The Guard with this validator
# guard_summary = Guard().use(
#     SaliencyCheck(
#         llm_callable="gpt-4o-mini",
#         threshold=0.1,
#         on_fail="exception",
#     )
# )
# Setup Guard
# guard_politness = Guard().use(
#     PolitenessCheck,
#     llm_callable="gpt-4o-mini",
#     on_fail="exception",
# )
input_gaurds =[guard_prompt]
output_gaurds = [ guard_PII]
# Use the Guard with the validator
guard_gibberish = Guard().use(
    GibberishText(
        threshold=0.5,
        validation_method="sentence",
        on_fail="exception",
    )
)

def check_input(input_text: str) -> bool:
    """Check the input text for safety using the defined guards."""
    for guard in gaurds:
        try:
            guard.validate(input_text)
        except Exception as e:
            print(f"Input validation failed: {e}")
            return False
    return True
def check_output(output_text: str) -> bool:
    """Check the output text for safety using the defined guards."""
    for guard in output_gaurds:
        try:
            guard.validate(output_text)
        except Exception as e:
            print(f"Output validation failed: {e}")
            return False
    return True


