from flask import Flask, request, render_template
from jinja2 import Template

app = Flask(__name__)
app.config['SECRET_KEY'] = 'THIS_HAS_TO_CHANGE!!!!!!!!!'

form = Template("""
<form method="{{method}}">
{% for form in input_forms %}
    {{form}}
{% endfor %}
    <input type="submit">
</form>
""")

tryit = Template("""
<div name="Form1> 
{{form1}}
</div>
<div name="Form2> 
{{form2}}
</div>
""")

input_form = Template("""
    <p>{{field.name}} = <input name="{{field.name}}" value="{{field.default}}"></p>
""")

select_form = Template("""
<select name={{name}}>
{% for opt_name in options %}
  <option value="{{opt_name}}">{{opt_name}}</option>
{% endfor %}
</select>
""")

class Field:
    def __init__(self, name):
        self.name = name
        self.default = name

def get_fields():
    return [Field(name) for name in ["hallo", "du", "arsch"]]

def prepare_form(fields):
    inputf = [input_form.render(field=field) for field in fields]
    return inputf + [select_form.render(name='try', options=['try', 'try1', 'try2'])]

@app.route('/')
def my_form():
    fields = get_fields()
    fields = prepare_form(fields)
    form1 = form.render(input_forms=fields, method="POST")
    form2 = form.render(input_forms=fields, method="POST2")
    return tryit.render(form1=form1, form2=form2)
    

def generate_route(app, route, method):

    def _wrapper():
        print(request.method)
        if request.method == method:
            fields = get_fields()
            fields = prepare_form(fields)
            form1 = form.render(input_forms=fields, method="POST")
            form2 = form.render(input_forms=fields, method="POST2")
        return tryit.render(form1=form1, form2=form2)

    # overwrite the name with some generic form
    _wrapper.__name__ = str(app) + f'-{method}'
    # 
    return app.route(route, methods=[method])(_wrapper)

generate_route(app, '/', 'POST')
generate_route(app, '/', 'POST2')


if __name__ == '__main__':
    app.run()
