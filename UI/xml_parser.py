try:
    import xml.etree.cElementTree as et
except ImportError:
    import xml.etree.ElementTree as et


class GeneratorObject():
    def __init__(self, args):
        self.id = args[0]
        self.model_name = args[1]
        self.transformation = args[2]
        self.object = args[3]
        self.object_type = args[4]
        self.location = args[5]
        self.pre_conditions = args[6]
        self.model_path = args[7]
        self.input_path = args[8]
        self.output_path = args[9]
        self.running_script = args[10]

    def update(self, args):
        self.model_name = args[0]
        self.transformation = args[1]
        self.object = args[2]
        # self.object_type = args[3]
        self.location = args[3]
        self.pre_conditions = args[4]
        self.model_path = args[5]
        self.input_path = args[6]
        self.output_path = args[7]
        self.running_script = args[8]


class ModelObject():
    def __init__(self, args):
        self.name = args[0]
        self.model_name = args[1]
        self.model_path = args[2]
        self.running_script = args[3]

class TransformationObject():
    def __init__(self, args):
        self.name = args[0]
        self.engine = args[1]
        self.params = args[2]
        self.running_script = args[3]


class ParamObject():
    def __init__(self, args):
        self.name = args[0]
        self.type = args[1]
        self.value = args[2]
        self.check = args[3]

class MUTObject():
    def __init__(self, args):
        self.name = args[0]
        self.class_name = args[1]
        self.path = args[2]
        self.running_script = args[3]

class OTCObject():
    def __init__(self, args):
        self.name = args[0]
        self.path = args[1]

def parse_new():
    parser = et.parse("../rmt_new.xml")
    root = parser.getroot()
    transformation_list = []
    model_list = []
    data_list = []
    for node in root.findall("transformation"):
        id = node.get("id")
        name = node.find("name").text
        engine = node.find("engine").text
        params = node.findall("param")
        running_script = node.find("running_script").text
        param_objs = []
        for param in params:
            obj = ParamObject([param.attrib['name'], param.attrib['type'], param.text, param.attrib['check']])
            param_objs.append(obj)
        transformation = TransformationObject([name, engine, param_objs, running_script])
        transformation_list.append(transformation)
    
    for node in root.findall("MUT"):
        name = node.find("name").text
        class_name = node.find("class_name").text
        path = node.find("path").text
        running_script = node.find("running_script").text
        MUT = MUTObject([name, class_name, path, running_script])
        model_list.append(MUT)

    for node in root.findall("OTC"):
        name = node.find("name").text
        path = node.find("path").text
        OTC = OTCObject([name, path])
        data_list.append(OTC)

    return transformation_list, model_list, data_list

def parse():
    parser = et.parse("../rmt.xml")
    root = parser.getroot()
    # print(root.tag)
    gen_list = []
    driving_model_list = []
    for name in root.findall("generator"):
        id = name.get("id")
        model_name = name.find("model_name").text
        transformation = name.find("transformation").text
        object = name.find("object").text
        object_type = name.find("object_type").text
        location = name.find("location").text
        pre_conditions = name.find("Pre_conditions").text
        model_path = name.find("model_path").text
        input_path = name.find("input_path").text
        output_path = name.find("output_path").text
        running_script = name.find("running_script").text
        gen = GeneratorObject([id, model_name, transformation, object, object_type,
                               location, pre_conditions, model_path, input_path, output_path, running_script])
        gen_list.append(gen)

    for name in root.findall("driving_model"):
        display_name = name.find("name").text
        model_name = name.find("model_name").text
        model_path = name.find("model_path").text
        running_script = name.find("running_script").text
        driving_model = ModelObject(
            [display_name, model_name, model_path, running_script])
        driving_model_list.append(driving_model)
    return gen_list, driving_model_list


def updateGenerator(generator):
    parser = et.parse("../rmt.xml")
    root = parser.getroot()
    for name in root.findall("generator"):
        id = name.get("id")
        if id == generator.id:
            name.find("model_name").text = generator.model_name
            name.find("transformation").text = generator.transformation
            name.find("object").text = generator.object
            # object_type = name.find("object_type").text
            name.find("location").text = generator.location
            name.find("Pre_conditions").text = generator.pre_conditions
            name.find("model_path").text = generator.model_path
            name.find("input_path").text = generator.input_path
            name.find("output_path").text = generator.output_path
            name.find("running_script").text = generator.running_script
            break
    parser.write("../rmt.xml")

def addGenerator(generator):
    parser = et.parse("../rmt.xml")
    root = parser.getroot()
    new_generator = et.Element('generator', {'id': generator.id})
    model_name= et.Element('model_name')
    model_name.text = generator.model_name
    new_generator.append(model_name)  

    transformation = et.Element('transformation')
    transformation.text = generator.transformation
    new_generator.append(transformation)  

    object= et.Element('object')
    object.text = generator.object
    new_generator.append(object)  

    object_type = et.Element('object_type')
    object_type.text = generator.object_type
    new_generator.append(object_type)  

    location= et.Element('location')
    location.text = generator.location
    new_generator.append(location)

    pre_conditions= et.Element('Pre_conditions')
    pre_conditions.text = generator.pre_conditions
    new_generator.append(pre_conditions)  

    model_path = et.Element('model_path')
    model_path.text = generator.model_path
    new_generator.append(model_path)  

    input_path = et.Element('input_path')
    input_path.text = generator.input_path
    new_generator.append(input_path)  

    output_path = et.Element('output_path')
    output_path.text = generator.output_path
    new_generator.append(output_path)  

    running_script = et.Element('running_script')
    running_script.text = generator.running_script
    new_generator.append(running_script)  

    root.append(new_generator)
    parser.write("../rmt.xml")

parse_new()
