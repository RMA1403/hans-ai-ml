from typing import List

def generate_prompt(
    calories: int,
    ingredients: List[str]
):
    ingredients_joined = ", ".join(ingredients)

    # TODO: Involve calories in the prompt
    prompt = f"""
    Berikan 1 resep yang terdiri dari (namun tidak terbatas pada) bahan-bahan berikut: {ingredients_joined}.

    Tuliskan bahan-bahan yang dibutuhkan beserta langkah-langkah pembuatannya.
    """

    return prompt
