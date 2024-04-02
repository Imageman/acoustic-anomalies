import json

class RecentItems:
    """Класс для хранения нескольких последних объектов"""
    def __init__(self, max_items):
        self.max_items = max_items
        if self.max_items<0: self.max_items=0
        self.items = []

    def __len__(self):
        return len(self.items)

    def add_item(self, item):
        self.items.append(item)
        while len(self.items) > self.max_items:
            self.items.pop(0)

    def get_item(self, index):
        return self.items[index]

    def get_items(self):
        return self.items

    def serialize(self):
        return json.dumps(self.items)

    def load(self, serialized):
        self.items = json.loads(serialized)
        while len(self.items) > self.max_items:
            self.items = self.items[-self.max_items:]

