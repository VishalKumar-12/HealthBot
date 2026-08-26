def format_medlineplus(results):

    if not results:
        return "No information found from MedlinePlus."

    context = ""

    for result in results:

        context += f"""
Title:
{result['title']}

Information:
{result['summary']}

Source:
{result['url']}

-------------------------
"""

    return context