from googlesearch import search
for item in search("как поймать воробья", lang="ru", num_results=10, advanced=True):
    print(f'url={item.url}, title={item.title}, description={item.description}\n')