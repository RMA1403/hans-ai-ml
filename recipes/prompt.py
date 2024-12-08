from typing import List

def generate_prompt(
    meal_type: str, # breakfast, lunch, dinner, snack
    calories: int,
    ingredients: List[str]
):
    ingredients_joined = ", ".join(ingredients)

    prompt = f"""
        Generate a recipe for a {meal_type} that is both healthy and ideally consists of (but is not restricted to) the following ingredients: {ingredients_joined}. The recipe should aim to provide approximately {calories} calories. Ensure the recipe includes:

        Ingredients list with specific quantities and substitutions (if applicable).
        1. Step-by-step instructions optimized for simplicity and clarity.
        2. Suggestions for making the recipe adaptable for dietary preferences or restrictions.
        3. Optional nutritional breakdown for key macros (e.g., protein, carbs, fats).

        Additional Notes: The recipe should prioritize wholesome, easy-to-find ingredients and techniques accessible to beginners but appealing enough for experienced cooks. If any ingredients are uncommon, propose widely available alternatives.
    """

    return prompt
