"""data_loader_compat.py — shared column constants (import shim)"""

NUMERIC_COLS = [
    'Caloric Value', 'Fat', 'Saturated Fats', 'Monounsaturated Fats',
    'Polyunsaturated Fats', 'Carbohydrates', 'Sugars', 'Protein',
    'Dietary Fiber', 'Cholesterol', 'Sodium', 'Water',
    'Vitamin A', 'Vitamin B1', 'Vitamin B11', 'Vitamin B12',
    'Vitamin B2', 'Vitamin B3', 'Vitamin B5', 'Vitamin B6',
    'Vitamin C', 'Vitamin D', 'Vitamin E', 'Vitamin K',
    'Calcium', 'Copper', 'Iron', 'Magnesium', 'Manganese',
    'Phosphorus', 'Potassium', 'Selenium', 'Zinc', 'Nutrition Density'
]
MACRO_COLS = ['Caloric Value', 'Fat', 'Carbohydrates', 'Protein', 'Dietary Fiber', 'Sugars']
VITAMIN_COLS = ['Vitamin A', 'Vitamin B1', 'Vitamin B2', 'Vitamin B3',
                'Vitamin B5', 'Vitamin B6', 'Vitamin B11', 'Vitamin B12',
                'Vitamin C', 'Vitamin D', 'Vitamin E', 'Vitamin K']
MINERAL_COLS = ['Calcium', 'Copper', 'Iron', 'Magnesium', 'Manganese',
                'Phosphorus', 'Potassium', 'Selenium', 'Zinc']
