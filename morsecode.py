def morse_code_converter(text, direction="encode"):
    """
    Converts text to Morse code or Morse code to text.

    Args:
        text (str): The input string to convert.
        direction (str): "encode" to convert text to Morse code,
                         "decode" to convert Morse code to text.
                         Defaults to "encode".

    Returns:
        str: The converted string.
    """

    MORSE_CODE_DICT = {
        'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.',
        'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---',
        'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---',
        'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-',
        'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--',
        'Z': '--..',
        '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-',
        '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.',
        ' ': '/',
        '.': '.-.-.-', ',': '--..--', '?': '..--..', "'": '.----.',
        '!': '-.-.--', '/': '-..-.', '(': '-.--.', ')': '-.--.-',
        '&': '.-...', ':': '---...', ';': '-.-.-.', '=': '-...-',
        '+': '.-.-.', '-': '-....-', '_': '..--.-', '"': '.-..-.',
        '$': '...-..-', '@': '.--.-.',
    }

    if direction == "encode":
        converted_message = []
        for char in text.upper():
            if char in MORSE_CODE_DICT:
                converted_message.append(MORSE_CODE_DICT[char])
            else:
                # Handle characters not in the dictionary (e.g., special symbols not listed)
                converted_message.append(char)
        return ' '.join(converted_message)
    elif direction == "decode":
        # Create a reverse dictionary for decoding
        DECODE_MORSE_CODE_DICT = {value: key for key, value in MORSE_CODE_DICT.items()}
        words = text.split(' / ')  # Split by word separator
        decoded_words = []
        for word in words:
            chars = word.split(' ') # Split into individual Morse code characters
            decoded_chars = []
            for char_code in chars:
                if char_code in DECODE_MORSE_CODE_DICT:
                    decoded_chars.append(DECODE_MORSE_CODE_DICT[char_code])
                else:
                    decoded_chars.append(char_code) # Keep unknown codes as is
            decoded_words.append(''.join(decoded_chars))
        return ' '.join(decoded_words)
    else:
        return "Invalid direction. Please choose 'encode' or 'decode'."

# --- Example Usage ---
if __name__ == "__main__":
    # Encode text to Morse code
    text_to_encode = "Hello World"
    encoded_message = morse_code_converter(text_to_encode, direction="encode")
    print(f"'{text_to_encode}' encoded to Morse code: {encoded_message}")

    print("-" * 30)

    # Decode Morse code to text
    morse_to_decode = ".... . .-.. .-.. --- / .-- --- .-. .-.. -.."
    decoded_message = morse_code_converter(morse_to_decode, direction="decode")
    print(f"'{morse_to_decode}' decoded to text: {decoded_message}")

    print("-" * 30)

    text_to_encode_2 = "Python is fun!"
    encoded_message_2 = morse_code_converter(text_to_encode_2, direction="encode")
    print(f"'{text_to_encode_2}' encoded to Morse code: {encoded_message_2}")

    print("-" * 30)

    morse_to_decode_2 = ".--. -.-- - .... --- -. / .. ... / ..-. ..- -. -.-.--"
    decoded_message_2 = morse_code_converter(morse_to_decode_2, direction="decode")
    print(f"'{morse_to_decode_2}' decoded to text: {decoded_message_2}")
