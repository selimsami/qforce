from .qform import QuestionForm, QuestionVisitor


class SettingsVistor(QuestionVisitor):
    """Get the settings for the javascript interface"""

    __slots__ = ()

    def visit_qform(self, qform, **kwargs):
        return qform.form.accept(self)

    def visit_question_block(self, block):
        out = {block.name: {'fields': {question.name: question.accept(self)
                                       for question in block.concrete.values()},
                            'previous': None}}
        #
        for subblock in block.blocks.values():
            out.update(subblock.accept(self))
        return out

    def visit_concrete_question_select(self, question):
        return {"type": "select",
                "label": question.label,
                "id": question.id,
                "value": question.answer,
                "is_set": question.is_set,
                "options": question.choices.as_list(),
                "is_optional": question.is_optional,
                "typ": question.typ,
                "comment": question.comment,
                }

    def visit_concrete_question_input(self, question):
        return {"type": "input",
                "label": question.label,
                "id": question.id,
                "value": question.answer,
                "is_set": question.is_set,
                "placeholder": question.placeholder,
                "is_optional": question.is_optional,
                "typ": question.typ,
                "comment": question.comment,
                }

    def visit_literal_block(self, block):
        value = block.get_answers()
        if value is None:
            value = ''
        return {"type": "literal",
                "label": block.label,
                "value": value,
                }


class ColtWebform(QuestionForm):
    """Colt questionform for webinterface"""
    # visitor to create json representation
    settings_visitor = SettingsVistor()

    def update_select(self, name, answer):
        """select"""
        out = {'delete': {}, 'setup': {}}
        #
        if answer == "":
            return out
        #
        block, key = self._split_keys(name)
        #
        if key in block.blocks:
            block = block.blocks[key]
            if block.answer == answer:
                return out
            out['delete'] = block.get_delete_blocks()
            #
            block.answer = answer
            out['setup'] = block.accept(self.settings_visitor)
        else:
            block.concrete[key].answer = answer
        return out

    def generate_setup(self, presets=None):
        if presets is not None:
            self.set_presets(presets)
        return self.settings_visitor.visit(self)
