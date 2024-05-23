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

"The “frontier mentality” articulated in the Odyssey is related to archaeologically identifiable manifestations of Greek colonization, reflecting the outlook of “proto-colonial” Greek traders, who needed to amass money and influence before they could launch an expedition aimed at creating a completely new settlement. Many Greeks from the eighth to sixth centuries were restless and on the move. Intrepid individuals left established communities in mainland Greece and on the western coast of Asia to form settlements far away, thus creating the distinctive map of ancient Greece, the connect-the-dots network of townships strung along so many coasts and islands of the Mediterranean and the Black Sea. Greek colonizing activity intensified in the seventh and sixth centuries BC and is inseparable from the developments outlined in the previous chapter, especially the increasing determination of less wealthy Greeks, at a time of scarce resources, to secure themselves economic independence and political self determination. The drive for independence—the rebellious quality in the Greek character—was in turn tied up with their developed sense of individuality. In this chapter, we learn how the Greeks transplanted their distinctive way of life—their gods, their songs, their vines, their drinking parties—to almost every corner of the Mediterranean and the Black Sea. We meet a remarkable number of Greeks with defined personalities and goals, who saw themselves not just as members of a colony or class but as important entities in their own right. Some, the founders of colonies and the “tyrants,” belong to new categories of leader, but other colorful individuals who wanted their names known to posterity included poets, athletes, mercenaries, priestesses, entrepreneurs, vase painters, and explorers. Some of their stories are interlinked, because individuals who desired fame would commission a famous poet such as Pindar to promote their reputation and even to invent for them a family tree that would trace their descent from one of the rugged individuals of myth, such as Heracles or another Argonaut. The age of colonization is also an age of Greek individualism. The proliferation of individual Greek communities can make the study of this period confusing. The confusion is exacerbated by the ancient Greeks’ habit of reapplying old names to the new settlements they founded—Heraclea, Megara, Naxos. But the proliferation is what made this period so important. The Greeks exponentially increased the number of communities they lived in and the ethnic groups with which they had contact. They expanded their shared horizons fundamentally and forever. "
    )
    pages = format_text(text)
    print_pages(pages)