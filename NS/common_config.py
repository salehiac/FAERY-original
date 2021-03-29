
def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return getinstance



@singleton 
class Config:
    def __init__(self, seed):
        self.seed=seed



"""This variable should be initialised as the single instance of CommonConfig by the __main___ script"""
config_=None





