import requests
import xml.etree.ElementTree as ET


MEDLINEPLUS_URL = "https://wsearch.nlm.nih.gov/ws/query"


def search_medlineplus(query, max_results=3):

    params = {
        "db": "healthTopics",
        "term": query,
        "retmax": max_results,
        "rettype": "brief"
    }

    try:

        response = requests.get(
            MEDLINEPLUS_URL,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        root = ET.fromstring(response.text)

        results = []

        for document in root.findall(".//document"):

            title = ""
            summary = ""
            url = document.attrib.get("url", "")

            for content in document.findall("content"):

                name = content.attrib.get("name")

                if name == "title":
                    title = "".join(
                        content.itertext()
                    ).strip()

                elif name == "FullSummary":
                    summary = "".join(
                        content.itertext()
                    ).strip()

            if title or summary:

                results.append({
                    "title": title,
                    "summary": summary,
                    "url": url
                })

        return results

    except Exception as e:

        print("MedlinePlus API Error:", e)

        return []