class QuestionForm {

    constructor (name, containerid, setup, fieldvalidation, selectupdate) {
        this.name = name;
        this.container = container;
        // store urls
        this.urls = {'setup': setup, 'fieldvalidation': fieldvalidation, 'selectupdate': selectupdate}
        this.blocks = {};
        this.form = this._setup(name, containerid);
        this.setup();
    }

    update(blocks, deleteblocks) {
        // newblocks is a dictionary, deleteblocks is a list
        // remove all not needed blocks from the dom
        for (var blockname in deleteblocks) {
            console.log(blockname);
            if (blockname in this.blocks) {
                console.log(blockname);
                this.blocks[blockname].remove();
                delete this.blocks[blockname];
            }
        }

        for (var blockname in blocks) {
            if (blockname in this.blocks) {
                continue;
            }
            const blockinfo = blocks[blockname];
            const block = this.create_block(blockname, blockinfo.fields);
            this.blocks[blockname] = block;
            // add after previous block
            const previous = blockinfo.previous;
            if (previous) {
                if (previous in this.blocks) {
                    insertAfter(block, this.block[blockinfo.previous]);
                } else {
                    this.form.appendChild(block);
                }
            } else {
                this.form.appendChild(block);
            }
        }
    }

    _setup (name, containerid) {
        const container = document.getElementById(containerid);
        if (container) {
            var form = document.createElement("form");
            add_attribute(form, "id", name);
            var sub = document.createElement("input");
            add_attribute(sub, "type", "submit");
            form.appendChild(sub);
            container.appendChild(form);
            return form;
        } else {
            console.log("cannot find container")
        }
    }

    create_block(blockname, fields) {
        // Create a Question Block
        const block = document.createElement("div");
        add_attribute(block, "class", "question_block_container");
        const name = document.createElement("h2");
        name.innerHTML = blockname;
        block.appendChild(name);
        var attr = document.createAttribute("id");
        attr.value = get_blockname(blockname);
        //
        for (var key in fields) {
            const opt = fields[key]; 
            var obj = null;
            switch (opt.type) {
                case "input":
                    obj = create_inputfield(opt.label, opt.id, opt.id, 
                                            opt.value, opt.placeholder);
                    // only add question object event handler 
                    var qobj = obj._question_object;
                    qobj.addEventListener("change", this.update_input_form(qobj));
                    break;
                case "select":
                    var selected = null;
                    if (opt.value === "") {
                        selected = "---";
                    } else {
                        selected = opt.value
                    }
                    obj = create_select_field(opt.label, opt.id, opt.id, opt.options, selected);
                    // only add question object event handler 
                    var qobj = obj._question_object;
                    qobj._question_selected = selected;
                    qobj.addEventListener("change", this.update_select_form(qobj));
                    break;
            }
            block.appendChild(obj);    
        }
        return block;
    }

    setup() {
        send_request({}, this.urls.setup, function(response) {
                this.update(response, {});
             }.bind(this));
    }

    update_input_form(obj) {
        return function() {
            const data = {value: obj.value, name: obj.id};
            send_request(data, this.urls.fieldvalidation, function (response) {
                if (response.answer) {
                    obj.style.border = "1px solid #ccc";
                } else {
                    obj.style.border = "2px solid red";
                }
            }.bind(this));
        }.bind(this)
    }

    update_select_form(obj) {
        return function() {
            //
            const data = {value: obj.value, name: obj.id};
            //
            if (obj.value !== "") {
                send_request(data, this.urls.selectupdate, function (response) {
                    this.update(response.setup, response.delete);
                }.bind(this));
                if (obj._question_selected === "---") {
                    obj.childNodes[0].remove();
                    obj._question_selected = obj.value;
                }
            }
        }.bind(this);
    }
}


function insertAfter(newNode, referenceNode) {
    referenceNode.parentNode.insertBefore(newNode, referenceNode.nextSibling);
}


function send_request(json, address, handle_response) {
    var xhr = new XMLHttpRequest();
    xhr.open("POST", address, true);
    xhr.setRequestHeader("Content-Type", "application/json");

    xhr.onreadystatechange = function() {
        if (xhr.readyState == 4) {
            handle_response(JSON.parse(xhr.responseText));
        }
    }
    xhr.send(JSON.stringify(json));
    return xhr.response;
}


function get_blockname(blockname) {
    return "questions::" + blockname;
}

function create_label(forobj, label) {
    var obj = document.createElement("value");
    add_attribute(obj, "for", forobj);
    obj.innerHTML = label;
    return obj;
}

function create_inputfield(label, id, name, value, placeholder) {
    var main = document.createElement("div");
    const label_obj = create_label(id, label);
    // 
    var obj = document.createElement("input");
    add_attribute(obj, "id", id);
    add_attribute(obj, "name", name);
    add_attribute(obj, "value", value);
    add_attribute(obj, "placeholder", placeholder);
    add_attribute(obj, "onkeydown", "return event.key != 'Enter';");
    //
    main.appendChild(label_obj);
    main.appendChild(obj);
    main._question_object = obj;
    return main;
}

function create_select_field(label, id, name, options, selected) {
    var main = document.createElement("div");
    const label_obj = create_label(id, label);
    //
    var obj = document.createElement("select");
    add_attribute(obj, "id", id);
    add_attribute(obj, "name", name);
    // ensure that none is always the first
    if (selected === "---") {
        // add default empty option
        var option = document.createElement("option");
        add_attribute(option, "value", "");
        option.innerHTML = "---";
        add_attribute(option, "selected", "selected");
        obj.appendChild(option);
    }
    // add keys
    for (var key in options) {
        var option = document.createElement("option");
        add_attribute(option, "value", options[key]);
        //
        if (key === selected) {
            add_attribute(option, "selected", "selected");
        }
        //
        option.innerHTML = options[key];
        obj.appendChild(option);
    }
    //
    main.appendChild(label_obj);
    main.appendChild(obj);
    main._question_object = obj;
    return main;
}

function add_attribute(obj, name, value) {
        var attr = document.createAttribute(name);
        attr.value = value;
        obj.setAttributeNode(attr);
}

function add_attribute(obj, name, value) {
        var attr = document.createAttribute(name);
        attr.value = value;
        obj.setAttributeNode(attr);
}

