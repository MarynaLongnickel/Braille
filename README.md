# Braille
OpenCV project that translates an image with braille into text.

Not very robust at the moment and can only translate braille that has the propper dimensions and alignment as specified in the US braille standards. (Some finagling with tolerance and image size might also be needed prior to translating.)

There are also different ways to express several symbols such as a period, and as of right now the code only accounts for one of them.
It also doesn't handle indicators to capitalize the entire word (such as `^^`) or numbers with more than one digit.

Also, if using an online [text-to-braille translator](https://www.atractor.pt/mat/matbr/matbraille-_en.html) to generate an image, it might be the case that the translator is using improper conversions (see the output from the code below, dot and apostrophe are represented by the same symbol in the image).

![braille](https://i.imgur.com/1Ox0UOg.jpg)
