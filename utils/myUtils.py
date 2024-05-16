import pickle, traceback
from datetime import datetime
import shutil

class MyUtils(object):
    def loadPkl(filename):
        try:
            file = open(filename, 'rb')
            data = pickle.load(file, encoding='latin1')
            file.close()
            return data
        except pickle.UnpicklingError as e:
            # normal, somewhat expected
            traceback.print_exc()
        except (AttributeError,  EOFError, ImportError, IndexError) as e:
            # secondary errors
            traceback.print_exc()
        except Exception as e:
            # everything else, possibly fatal
            traceback.print_exc()
            return

    def dumpPkl(data, filename):
        try:
            file = open(filename, 'wb')
            data = pickle.dump(data, file)
            file.close()
            print("saved in {}".format(filename))
        except Exception as e:
            # everything else, possibly fatal
            traceback.print_exc()
            return
    def unitNormalize(arr):
        norm = np.max(arr) - np.min(arr)
        if norm == 0: 
            return arr
        else: 
            return (arr - np.min(arr))/norm
        return
    
    def dumpPklWithTimestamp(data, filename):
        if len(filename.split('.')) == 2:
            fname, suffix = filename.split('.')
            outFname = "{}_{}.{}".format(fname, MyUtils.getTimestamp(), suffix)
        else:
            outFname = "{}_{}".format(filename, MyUtils.getTimestamp())
        MyUtils.dumpPkl(data, outFname)
                                      
    def getTimestamp():
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d_%H%M%S")
        return dt_string
    
    def copyFile(inFname, outFname):
        shutil.copyfile(inFname, outFname)
        print("copy file {} to {}".format(inFname, outFname))

    def copyFileWithTimestamp(filename, label = "", TIME = True):
        if len(filename.split('.')) == 2:
            fname, suffix = filename.split('.')
            suffix =  "." + suffix
        else:
            fname, suffix = filename, ""
        
        outFname = "{}_{}{}{}".format(fname, MyUtils.getTimestamp() if TIME else "", label, suffix)
        MyUtils.copyFile(filename, outFname) 
