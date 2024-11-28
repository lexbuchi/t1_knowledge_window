# utils.py

from hashlib import md5
from jinja2 import Template

def encode_md5(text):
    return md5(text.encode("utf-8")).hexdigest()

def construct_prompt(template_jinja, bos_token, eos_token, task_instruction, parser_instruction, additional_inst, pre_query, query):
    messages = [
        {'role': 'system', 'content': f"{task_instruction}"},
        {'role': 'user', 'content': f"{parser_instruction}{additional_inst}\n{pre_query} {query}" }
    ]
    add_generation_prompt = True
    template = Template(template_jinja)
    rendered_template = template.render(
        messages=messages,
        bos_token=bos_token,
        eos_token=eos_token,
        add_generation_prompt=add_generation_prompt
    )
    return rendered_template