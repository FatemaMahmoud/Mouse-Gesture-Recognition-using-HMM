import tkinter as tk
from tkinter import messagebox

class HmmFrame(tk.Frame):  
    
    points = list()
    classifier = None
    def __init__(self, master=None):
        tk.Frame.__init__(self, master, background = "#930138", width=650, height=610)
        self.master.title('Hidden Markov Model Application')
        self.place(x = 800, y = 200)
        self.master.resizable(0, 0)
        self.pack(fill="both", expand=1)
        self.create_widgets()
    
    def classifier(self, classifier):
        self.classifier = classifier
        
    def create_widgets(self):
        self.drawingCanvas = tk.Canvas(self, width=630, height=500, background="white", highlightbackground="#e8e3e3", highlightthickness = 5 )
        self.drawingCanvas.place(x=5, y=5)
        self.drawingCanvas.bind('<Button-1>', self.on_click)
        self.drawingCanvas.bind('<B1-Motion>', self.on_motion)
        self.drawingCanvas.bind('<ButtonRelease-1>', self.on_release)
        
        
        self.drawLabel = tk.Label(self,  background = "white", foreground = "#930138", text = "Draw something!", font=("Arial Narrow", 16, "bold") )
        self.drawLabel.place(x=220, y=10, width= 200, height =35)
        
        self.clearButton = tk.Button(self, background = "white", foreground = "#930138", text = "Clear", font=("Arial Narrow", 16, "bold"))
        self.clearButton.place(x=280, y=520, width= 80, height =30)
        self.clearButton.bind('<Button-1>', self.on_click_clear)
        
        
        
    def on_click(self, event):
        self.points.append([event.x, event.y])
        
    def on_motion(self, event):
        self.points.append([event.x, event.y])
        self.line = self.drawingCanvas.create_line(self.points , fill = "black", width = 3)
        
    def on_release(self, event):
        tk.messagebox.askquestion("Sequence Class", self.classifier.classify(self.points))
        self.points.clear()
        
    def on_click_clear(self, event):
        self.drawingCanvas.delete("all")