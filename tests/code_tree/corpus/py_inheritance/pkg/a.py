class Base:
    def run(self):
        return 1


class Other:
    def run(self):
        return 2


class Derived(Base):
    def caller(self):
        return self.run()
