from creator import CustomStructureCreator


class JoinCreators(CustomStructureCreator):
    """Join Computation creators to execute them at the same time"""

    def __init__(self, *creators):
        self._creators = creators

    def setup_pre(self, qm):
        for creator in self._creators:
            creator.setup_pre(qm)

    def check_pre(self):
        for creator in self._creators:
            cal = creator.check_pre()
            if cal is not None:
                return cal
        return None

    def parse_pre(self, qm):
        for creator in self._creators:
            creator.parse_pre(qm)

    def setup_main(self, qm):
        for creator in self._creators:
            creator.setup_main(qm)

    def check_main(self):
        for creator in self._creators:
            cal = creator.check_main()
            if cal is not None:
                return cal
        return None

    def parse_main(self, qm):
        for creator in self._creators:
            creator.parse_main(qm)

    def setup_post(self, qm):
        for creator in self._creators:
            creator.setup_post(qm)

    def check_post(self):
        for creator in self._creators:
            cal = creator.check_post()
            if cal is not None:
                return cal
        return None

    def parse_post(self, qm):
        for creator in self._creators:
            creator.parse_post(qm)
