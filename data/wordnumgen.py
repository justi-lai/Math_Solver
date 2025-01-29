import csv
import random
import inflect
from fractions import Fraction
import os

# Initialize inflect engine for converting numbers to words
inflect_engine = inflect.engine()

def number_to_words(number):
    """Convert a number to its word form."""
    if isinstance(number, Fraction):
        numerator = inflect_engine.number_to_words(number.numerator)
        denominator = inflect_engine.number_to_words(number.denominator)
        return f"{numerator} over {denominator}"
    elif isinstance(number, float):
        integer_part = int(number)
        decimal_part = str(number).split(".")[1]
        integer_words = inflect_engine.number_to_words(integer_part)
        decimal_words = " ".join([inflect_engine.number_to_words(int(d)) for d in decimal_part])
        return f"{integer_words} point {decimal_words}"
    elif isinstance(number, str) and number.endswith("%"):
        base_number = float(number.strip('%'))
        return f"{inflect_engine.number_to_words(base_number)} percent"
    else:
        return inflect_engine.number_to_words(number)

def generate_random_numbers(count=10):
    """Generate random numbers and convert them to word form."""
    results = []
    for _ in range(count):
        choice = random.choice(["integer", "decimal", "fraction", "percentage", "numerical_int", "numerical_float", "numerical_fraction", "numerical_percentage"])
        if choice == "integer":
            number = random.randint(1, 1000)
            results.append((number, number_to_words(number)))
        elif choice == "decimal":
            number = round(random.uniform(1, 100), 2)
            results.append((number, number_to_words(number)))
        elif choice == "fraction":
            number = Fraction(random.randint(1, 10), random.randint(1, 10))
            decimal_number = float(number.numerator) / float(number.denominator)
            results.append((decimal_number, number_to_words(number)))
        elif choice == "percentage":
            number = f"{random.randint(1, 100)}%"
            results.append((number, number_to_words(number)))
        elif choice == "numerical":
            number = random.randint(1, 1000)
            results.append((number, number))
        elif choice == "numerical_float":
            number = round(random.uniform(1, 100), 2)
            results.append((number, number))
        elif choice == "numerical_fraction":
            number = Fraction(random.randint(1, 10), random.randint(1, 10))
            decimal_number = float(number.numerator) / float(number.denominator)
            results.append((decimal_number, number_to_words(number)))
        elif choice == "numerical_percentage":
            number = f"{random.randint(1, 100)}%"
            results.append((number, number))
        else:
            continue
        
    return results

def save_to_csv(data, filename="random_numbers.csv"):
    """Save the generated data to a CSV file in the same directory as the script."""
    # Get the directory of the current script
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # Create the full file path
    file_path = os.path.join(script_directory, filename)
    
    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["Number", "Words"])
        # Write the data
        writer.writerows(data)
    
    print(f"Data successfully saved to {file_path}")

# Example usage
if __name__ == "__main__":
    generated_numbers = generate_random_numbers(count=10000)
    save_to_csv(generated_numbers)
    for number, words in generated_numbers:
        print(f"Number: {number} -> Words: {words}")
