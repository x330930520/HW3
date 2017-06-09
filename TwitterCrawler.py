#coding:utf-8
import json
import math
import tweepy 
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler, Stream
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import *
import prettytable
from prettytable import PrettyTable
from mpl_toolkits.basemap import Basemap
from geopy import geocoders
    
consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


def homepage():
    public_tweets = api.home_timeline()
    hp = ""
    #Pritn out content on the homepage 
    for tweet in public_tweets[:15]:
        hp += '\n'
        hp += tweet.text
        hp += '\n'
        hp += '\n'
    return hp

class TwitterStream(StreamListener):
    def on_data(self, data):
        try:
            data = json.loads(data)
            create = str(data['created_at'])
            id = str(data['user']['id'])
            name = str(data['user']['name'])
            location = str(data['user']['location'])
            lang = str(data['user']['lang'])
            text_msg = str(data['text'])
            with open(("Geo_data.csv"), "a", encoding='utf-8') as f:
                if data['coordinates'] == None :
                    pass
                else:
                    coord = str(data['coordinates']['coordinates'])
                    lng = coord[:coord.rfind(',')].replace("[","")
                    lat = coord.split(" ")[1].replace("]","")
                    string = lng + "," + lat
                    #string = string.encode('ascii', 'ignore')
                    #try:
                    print(str(string), file=f)
                    #except UnicodeEncodeError or IndexError:
                    #string.encode('ascii', 'ignore')
            with open('tweets.csv', 'a', encoding='utf-8') as fi:
                    tweets = create + "," + name + "," + id + "," + text_msg
                    tweets = tweets.replace('\n', ' ')
                    tweets = tweets.encode('ascii', 'ignore')
                    #try:
                    print (str(tweets))
                    print (str(tweets), file=fi)
                    #except KeyError or IndexError or UnicodeEncodeError:
                        #tweet.encode('ascii', 'ignore')
            with open('raw_tweets.csv', 'a', encoding='utf-8') as fil:
                print(data, file=fil)
        except KeyError or IndexError or UnicodeEncodeError:
            pass
        return True

    def on_error(self, status):
        print(status)

def live(ask):
    streaming = TwitterStream()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    stream = Stream(auth, streaming, timeout = 5000)
    stream.filter(track = ask)
    
def trend(woe_id): #updated every 5 minutes
    api = tweepy.API(auth)
    t2 = api.trends_place(woe_id)
    table = PrettyTable(field_names=['Count', 'Trend'])
    
    for x in t2[0]['trends'][:15]:
        table.add_row([x['tweet_volume'], x['name']])
    #print (table)
    return table


    
class Geo_listener(StreamListener):
    def on_data(self, data):
        data = json.loads(data)
        with open(("Geo_loc.csv"), "a") as f:
            id = str(data['user']['id'])
            location = str(data['user']['location'])
            lang = str(data['user']['lang'])
            text_msg = str(data['text'])
            if data['coordinates'] == None :
                info = 'No geo information for this tweet'
                tweets = info + "," + id + "," + location + "," + text_msg
                tweets = tweets.replace('\n', ' ')
                lng = ""
                lat = ""
            else:
                coord = str(data['coordinates']['coordinates'])
                lng = coord[:coord.rfind(',')].replace("[","")
                lat = coord.split(" ")[1].replace("]","")

                tweets = lng + "," + lat + ","  + id + "," + "," + lang + "," + "," + text_msg
                tweets = tweets.replace('\n', ' ')
                tweets = tweets.encode('ascii', 'ignore')
                try:
                    print(str(tweets))
                    print(lng + "," + lat, file=f)
                except UnicodeEncodeError:
                    pass
        return True

    def on_error(self, statut):
        print(statut)


def deg2rad(degrees):
    return math.pi*degrees/180.0

def rad2deg(radians):
    return 180.0*radians/math.pi



# According to WGS-84 
def w_84(lat):
    w_84_a = 6378137.0  
    w_84_if = 298.257223563
    w_84_f = 1/w_84_if
    w_84_b = (w_84_a*(1-w_84_f)) 
    An = w_84_a*w_84_a * math.cos(lat)
    Bn = w_84_b*w_84_b * math.sin(lat)
    Ad = w_84_a * math.cos(lat)
    Bd = w_84_b * math.sin(lat)
    return math.sqrt((An*An + Bn*Bn)/(Ad*Ad + Bd*Bd))


def bound(lat_deg, ing_deg, half_km):
    lat = deg2rad(lat_deg)
    lng = deg2rad(ing_deg)
    half = 1000*half_km

    radius = w_84(lat)
    
    pradius = radius*math.cos(lat)

    latMin = lat - half/radius
    latMax = lat + half/radius
    lngMin = lng - half/pradius
    lngMax = lng + half/pradius

    return [rad2deg(latMin), rad2deg(lngMin), rad2deg(latMax), rad2deg(lngMax)]

def start_geo_listener(address, rad):
    gc = geocoders.GoogleV3()
    place, (lat, lng) = gc.geocode(address)
    loc = [bound(lng,lat,rad)[0],bound(lng,lat,rad)[1],bound(lng,lat,rad)[2],bound(lng,lat,rad)[3]]
    twitterStream = Stream(auth, Geo_listener(), timeout = 5000)
    twitterStream.filter(locations = loc)
    
    
def draw():
    x=[]
    y=[]
    z=[]
    with open('Geo_data.csv') as f:
            for line in f:
                x.append(float(line.split(',')[0]))
                y.append(float(line.split(',')[1]))
    m = Basemap(projection='mill',lon_0=-50,lat_0=60,resolution='l')
        
    x1,y1=m(x,y)
    m.drawcoastlines()
    m.drawmapboundary(fill_color='grey') 
    m.drawcountries()
    m.fillcontinents(color='white',zorder=0)
    m.scatter(x1,y1,color = 'red',alpha=0.3)
    plt.title("Twitter User Location with Basemap")
    plt.show()
    #return m

    
root = tk.Tk
class GUI(root):

    def __init__(self, *args, **kwargs):
        root.__init__(self, *args, **kwargs)
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for page in (Homepage, HomePage, Trend, US, RU, CA, IN, UK, AU, Live_Key, Live_Locaion, Map):
            page_name = page.__name__
            frame = page(container, self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("Homepage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()


class Homepage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text = "Homepage", font = ("Helvetica", 18, "bold"))
        label.pack(side="top", fill = "x", pady=10)

        button1 = tk.Button(self, text = "Show My Home Page", command = lambda: controller.show_frame("HomePage"))
        button2 = tk.Button(self, text = "Show World Trend", command = lambda: controller.show_frame("Trend"))
        button3 = tk.Button(self, text = "Live Stream with Key Word", command = lambda: controller.show_frame("Live_Key"))
        button4 = tk.Button(self, text = "Live Stream with Location", command = lambda: controller.show_frame("Live_Locaion"))
        button5 = tk.Button(self, text = "Map Locations", command = lambda: controller.show_frame("Map"))
        button1.pack( pady = 15)
        button2.pack( pady = 15)
        button3.pack( pady = 15)
        button4.pack( pady = 15)
        button5.pack( pady = 15)


class HomePage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text = "Show My Home Page", font=("Helvetica", 18, "bold"))
        label.pack(side = "top", fill = "x", pady=10)
        button = tk.Button(self, text = "Go to Homepage", bg = 'green', command = lambda: controller.show_frame("Homepage"))
        text = Message (self, text = homepage(), justify = LEFT)
        button.pack(pady  = 15)
        text.pack(pady = 15)

        
class Trend(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text = "Show World Trend", font=("Helvetica", 18, "bold"))
        label.pack(side = "top", fill = "x", pady=10)
        button1 = tk.Button(self, text = "Go to Homepage", bg = 'green', command = lambda: controller.show_frame("Homepage"))
        button2 = tk.Button(self, text = "USA", command = lambda: controller.show_frame("US"))
        button3 = tk.Button(self, text = "Russia", command = lambda: controller.show_frame("RU"))
        button4 = tk.Button(self, text = "Canada", command = lambda: controller.show_frame("CA"))
        button5 = tk.Button(self, text = "India", command = lambda: controller.show_frame("IN"))
        button6 = tk.Button(self, text = "UK", command = lambda: controller.show_frame("UK"))
        button7 = tk.Button(self, text = "Australia", command = lambda: controller.show_frame("AU"))
        button1.pack(  pady = 15)
        button2.pack(  padx = 60, pady = 15)
        button3.pack(  padx = 60, pady = 15)
        button4.pack(  padx = 60, pady = 15)
        button5.pack(  padx = 60, pady = 15)
        button6.pack(  padx = 60, pady = 15)
        button7.pack(  padx = 60, pady = 15)

        
class US(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text = "USA Trend", font=("Helvetica", 18, "bold"))
        label.pack(side = "top", fill = "x", pady=10)
        button = tk.Button(self, text = "Go to Homepage", bg = 'green', command = lambda: controller.show_frame("Homepage"))
        text = Message (self, text = trend(23424977), justify = CENTER)
        button.pack( pady = 15)
        text.pack( pady = 15)
        
class RU(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text = "Russia Trend", font=("Helvetica", 18, "bold"))
        label.pack(side = "top", fill = "x", pady=10)
        button = tk.Button(self, text = "Go to Homepage", bg = 'green', command = lambda: controller.show_frame("Homepage"))
        text = Message (self, text = trend(23424936), justify = CENTER)
        button.pack( pady = 15)
        text.pack( pady = 15)
        
class CA(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text = "Canada Trend", font=("Helvetica", 18, "bold"))
        label.pack(side = "top", fill = "x", pady=10)
        button = tk.Button(self, text = "Go to Homepage", bg = 'green', command = lambda: controller.show_frame("Homepage"))
        text = Message (self, text = trend(23424775), justify = CENTER)
        button.pack( pady = 15)
        text.pack( pady = 15)
        
class IN(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text = "India Trend", font=("Helvetica", 18, "bold"))
        label.pack(side = "top", fill = "x", pady=10)
        button = tk.Button(self, text = "Go to Homepage", bg = 'green', command = lambda: controller.show_frame("Homepage"))
        text = Message (self, text = trend(23424848), justify = CENTER)
        button.pack( pady = 15)
        text.pack( pady = 15)
        
class UK(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text = "UK Trend", font=("Helvetica", 18, "bold"))
        label.pack(side = "top", fill = "x", pady=10)
        button = tk.Button(self, text = "Go to Homepage", bg = 'green', command = lambda: controller.show_frame("Homepage"))
        text = Message (self, text = trend(23424975), justify = CENTER)
        button.pack( pady = 15)
        text.pack( pady = 15)
        
class AU(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text = "Australia Trend", font=("Helvetica", 18, "bold"))
        label.pack(side = "top", fill = "x", pady=10)
        button = tk.Button(self, text = "Go to Homepage", bg = 'green', command = lambda: controller.show_frame("Homepage"))
        text = Message (self, text = trend(23424748), justify = CENTER)
        button.pack( pady = 15)
        text.pack( pady = 15)
        
class Live_Key(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Live Stream with Key Word", font=("Helvetica", 18, "bold"))
        label.pack(side="top", fill="x", pady=10)
        button1 = tk.Button(self, text="Go to Homepage", command = lambda: controller.show_frame("Homepage"))
        self.word = Entry(self)
        button2 = tk.Button(self, text="OK", bg = 'red', command = lambda: live(self.word.get()))
        button1.pack( pady = 15)
        self.word.pack( pady = 15)
        button2.pack( pady = 15)
        
        
class Live_Locaion(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Live Stream with Location", font=("Helvetica", 18, "bold"))
        label.pack(side="top", fill="x", pady=10)
        label1 = tk.Label(self, text="Enter a city name", font=("Helvetica", 12, "bold"))
        self.place = Entry(self)
        label2 = tk.Label(self, text="Enter a radius in KM", font=("Helvetica", 12, "bold"))
        self.rad = Entry(self)
        button1 = tk.Button(self, text="Go to Homepage", bg = 'green', command = lambda: controller.show_frame("Homepage"))
        button2 = tk.Button(self, text="OK", bg = 'red', command = lambda: start_geo_listener(self.place.get(), int(self.rad.get())))
        button1.pack( pady = 15)
        label1.pack( pady = 15)
        self.place.pack( pady = 15)
        label2.pack( pady = 15)
        self.rad.pack( pady = 15)
        button2.pack( pady = 15)
        
class Map(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Map Locations", font=("Helvetica", 18, "bold"))
        label.pack(side="top", fill="x", pady=10)
        button1 = tk.Button(self, text="Go to Homepage", bg = 'green',  command = lambda: controller.show_frame("Homepage"))
        button2 = tk.Button(self, text="Get Map", bg = 'red', command = lambda: draw())
        button1.pack( pady = 15)
        button2.pack( pady = 15)


#if __name__ == "__main__":
gui = GUI()
gui.mainloop()
    
    
    
