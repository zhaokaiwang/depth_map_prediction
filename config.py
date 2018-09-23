import configparser

def get_config(section, key):
    config = configparser.ConfigParser()
    path = 'configure.conf'
    
    config.read(path)
    return config.get(section, key)

if __name__ == '__main__':
    temp = get_config('NyuV2', 'input_height')
    pass