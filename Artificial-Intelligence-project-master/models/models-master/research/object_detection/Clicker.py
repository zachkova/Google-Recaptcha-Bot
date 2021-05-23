import Object_detection_image
from pynput.mouse import Button, Controller


class Clicker:

    def __init__(self):
        self.coords = None
        self.checkboxCoords = None

    def getBoxCoords(self):
        self.coords = Object_detection_image.recieve_object()
        print(self.coords)

    def findCheckBox(self):
        if self.coords is None:
            return

        for i in range(len(self.coords)):
            self.checkboxCoords = (self.coords[i][0] + 70, self.coords[i][2] + 70)
            print(self.checkboxCoords)
            self.clickBox()

    def clickBox(self):
        if self.checkboxCoords is None:
            return

        mouse = Controller()
        mouse.position = self.checkboxCoords
        mouse.click(Button.left, 1)

if __name__ == "__main__":
    clicker = Clicker()
    clicker.getBoxCoords()
    clicker.findCheckBox()