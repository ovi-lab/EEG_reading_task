def format_text(text, line_length=38, lines_per_page=9):
    words = text.split()
    pages = []
    current_page = []
    current_line = ""

    for word in words:
        # Check if adding the next word would exceed the line length
        if len(current_line) + len(word) + 1 <= line_length:
            if current_line:
                current_line += " "
            current_line += word
        else:
            # Add the current line to the current page
            current_page.append(current_line)
            current_line = word
            # Check if the current page has enough lines
            if len(current_page) == lines_per_page:
                pages.append(current_page)
                current_page = []

    # Add any remaining text to the current page
    if current_line:
        current_page.append(current_line)
    if current_page:
        pages.append(current_page)

    return pages

def print_pages(pages):
    for page_num, page in enumerate(pages):
        print(f"Page {page_num + 1}:")
        for line in page:
            print(line)
        print()

if __name__ == "__main__":
    text = (

"There were three hereditary classes in Spartan society. The superior or citizen class were the Spartiates. There was no distinction of status within this group—they also called themselves the homoioi, meaning “equals.” The homoioi were, however, subdivided into tribes. They lived together in the “city,” which consisted of five villages. The status of Spartiate needed to be inherited, consolidated through participation in the rigorous training of the agoge and maintained by membership in a mess and suitable conduct in battle. If these requirements were not met, the Spartiate could lose his status, suffer public humiliation, and have all his contracts with Spartiates annulled. The social system at Sparta, however peculiar, was probably a response to the same kind of crisis over land distribution that led to tyrannies and subsequently democracies in some other Greek cities, but it took a different form: the creation of a sense of solidarity and peer-group similarity among the ruling class. The group’s solidarity was reinforced by a complex system of power distribution within it. There were always supposed to be two kings—constituting a dyarchy rather than a monarchy. This was intended to prevent either king from accumulating excessive authority. The two Peloponnesian kings Menelaus and Agamemnon may be a mythical reflection of this constitutional peculiarity, although in reality the dyarchs were not brothers. They inherited their position as members of two distinct dynasties, the Agids and the Eurypontids, both of which claimed direct descent from Heracles, son of Zeus. When not at war, the kings’ main role was religious; they were both priests of Zeus. They swore reciprocal oaths with the ephors, the five annually elected officers of the Spartan citizen Assembly, vowing to respect each other’s authority: The lesson most taken to heart at Sparta, according to numerous sources, was “to rule and likewise to be ruled.” The ephors appointed magistrates and scrutinized their performance, although the Assembly voted on whether to go to war. To outsiders like Herodotus and Xenophon, however, the most remarkable feature of classical Sparta was the power of the older men. It was a gerontocracy. Herodotus is surprised that strong young Spartans always make way in the street for senior citizens, and there may have been a rule that Spartan youths had to give up their seats to their elders. The name of the council at Sparta, gerousia, even translates as something like “House of Elders.” Only men over sixty were eligible to become one of the twenty-eight gerontes (plus the two kings) who formed this body. The gerontes were the judges in capital offenses."
    
    )
    pages = format_text(text)
    print_pages(pages)