# Braille
OpenCV project that translates an image with braille into text __using edge detection__.

Not very robust at the moment and can only translate braille that has the propper dimensions and alignment as specified in the US braille standards. (Some finagling with tolerance and image size might also be needed prior to translating.)

There are also different ways to express several symbols such as a period, and as of right now the code only accounts for one of them.
It also doesn't handle indicators to capitalize the entire word (such as `^^`) or numbers with more than one digit.

Also, if using an online [text-to-braille translator](https://www.atractor.pt/mat/matbr/matbraille-_en.html) to generate an image, it might be the case that the translator is using improper conversions (the one in the link uses the same symbol for period and aphostrophe for example).

Below is an example of how this braille to text translator works.
![braille](https://i.imgur.com/PO8PzZV.jpg)
