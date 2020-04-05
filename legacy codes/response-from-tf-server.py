import urllib2
import json
import time

READ_API_KEY = '3OBYEMTGZUT0T5ZJXYJ0AA'
data = 100.0

# get data from cloud
def getdata():
    connt_speak = urllib2.urlopen("http://192.168.26.18:5000/update?api_key=%s&field=%s" % (READ_API_KEY, data))
    response = connt_speak.read()
    data_got = json.loads(response)
    print(data_got['data'])
    connt_speak.close()



def main():
   getdata()



if __name__  == '__main__':
  main()
