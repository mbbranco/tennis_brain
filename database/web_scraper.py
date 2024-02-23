from bs4 import BeautifulSoup
from datetime import date,timedelta
import urllib.request

def get_img_link(link):
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
    urllib.request.install_opener(opener)

    info = opener.open(link).read()
    soup = BeautifulSoup(info, "html.parser")
    i = soup.find(class_='player_image')

    if i is None:
        i = soup.find(class_=' player_image--headshot')

    img_link = 'https://www.atptour.com' + i.img['src']

    return img_link

def get_current_ranking_photo(p1_name,p2_name):
    players,links = get_updated_ranking()
    p1_link,p2_link = None,None

    if p1_name in players:
        p1_index = players.index(p1_name)
        p1_link = get_img_link(links[p1_index])

    if p2_name in players:
        p2_index = players.index(p2_name)
        p2_link = get_img_link(links[p2_index])

    return p1_link, p2_link


def get_updated_ranking():
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
    urllib.request.install_opener(opener)

    players_ranking = []
    links_ranking = []
    rank_range = ['0-5000']
    for r in rank_range:
        url = f'https://www.atptour.com/en/rankings/singles?rankRange={r}'
        info = opener.open(url).read()
        soup = BeautifulSoup(info, "html.parser")
        players, links = scrape_it(soup)

        players_ranking.extend(players)
        links_ranking.extend(links)

    return players_ranking, links_ranking

def scrape_it(soup):
    names = soup.find_all(class_='name center')

    players = []
    links = []
    for name in names:
        p = name.find('span').contents[0].strip()
        link = name.find('a')['href']

        players.append(p)
        links.append('https://www.atptour.com' + link)

    return players,links


if __name__=='__main__':
    p1_name = "Cameron Norrie"
    p2_name = "Carlos Alcaraz"

    p1_link,p2_link = get_current_ranking_photo(p1_name,p2_name)
    print(p1_link,p2_link)