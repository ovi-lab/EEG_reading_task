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

"Let me turn now to the ephors. These, according to Spartan tradition, were invented somewhat late in the development of the Spartan constitution. The word ephor comes from the word which means to oversee, to oversee what’s going on. They were, in a certain sense, the overseers. One of their duties was to keep watch on the kings and to see that the kings didn’t do anything improper, illegal, irreligious, or anything of that kind, and some scholars have focused on that and suggested that, at least originally, that was what their main function was: to protect the Spartans from excessive power, excessive behavior by the kings, and that their sort of watching the king’s thing was always their chief function. That, I think, is not right.  I think by the time the Spartans appear to us in history, let us say late in the sixth century and fifth century, the ephors don’t do that. I mean, they still have the technical constitutional requirement to do that, but that’s not what they’re up too. When we see them they are usually engaged in dealing with foreign policy. So, if a neighboring state wanted to communicate something to the Spartans, either it might be an offer of an alliance or it might be an order to do something or else war would follow, or a negotiation for peace, any of those things, first they would come to the ephors, of which there were five and the ephors would then decide what should be done.  I would say, in most cases, they would, unless it was very, very serious, they would be able to give some sort of answer to it, but when it involved something fundamental like war and peace or alliances, then they would have to go to the assembly to get their approval. But my guess is that it would have been wildly reckless and therefore never done for the ephors not to go to the gerousia first, because the gerousia was, by far, the most significant council in the state, most able to have the necessary prestige and yet to be small enough truly to discuss what needed to be done. And since thegerousia included the kings, it involved the most important people in the state. So, if the ephors wanted to do something, it would be damn foolish not to clear it with the gerousia first; although if they wished to be reckless, they could do otherwise.   Now, another thing about the ephors is that they’re very different. The people who are elected to the gerousia are old men who have proven themselves, they are truly elected by a process in which their individual qualities are relevant, and so they have tremendous prestige in the Spartan state. This is not true necessarily and typically of the ephors. Aristotle tells us that they in fact were just any Joe Spartan, that they were ordinary people, not distinguished in any way. "

    )
    pages = format_text(text)
    print_pages(pages)