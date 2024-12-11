from typing import List

def generate_prompt(
    calories: int,
    ingredients: List[str]
):
    ingredients_joined = ", ".join(ingredients)

    # TODO: Involve calories in the prompt
    prompt = '''Berikan saya satu resep beserta langkah-langkahnya yang dapat dibuat dari bahan-bahan berikut. Bisa saja terdapat beberapa bahan tambahan yang tidak tercantum dalam daftar bahan.
    {INGREDIENTS}

    Berikan dalam format
    ## JUDUL ###
    {Judul Makanan}
    ## KARBOHIDRAT ##
    {Karbohidrat} gram
    ## PROTEIN ##
    {Protein} gram
    ## LEMAK ##
    {Lemak} gram
    ## BAHAN ##
    {Bahan dipisahkan per baris}
    ## LANGKAH ##
    {Langkah-langkah dipisahkan per baris}
    '''
    prompt = prompt.replace("{INGREDIENTS}", ingredients_joined)

    return prompt
