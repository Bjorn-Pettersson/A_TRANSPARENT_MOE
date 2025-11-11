class TextCleaner:
    def __init__(self, data):
        self.raw_data = data
        self.processed = None

    def to_lowercase(self):
        # Converts all text to lowercase
        self.processed = self.processed.lower()
        return True
    def to_lowercase(self):
        # Converts all text to lowercase
        self.processed = self.processed.lower()
        return True

    def remove_punctuation(self):
    def to_lowercase(self):
        # Converts all text to lowercase
        self.processed = self.processed.lower()
        return True

    def to_lowercase(self):
        """
        Removes all common punctuation characters.
        This includes periods, commas, and question marks.
        """
        import string
        clean_data = self.raw_data.translate(str.maketrans('', '', string.punctuation))
        self.processed = clean_data
        return True

        # Converts all text to lowercase
        self.processed = self.processed.lower()
        return True