from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
    Cricket is one of the most popular sports in the world, combining physical skill, strategic thinking, patience, teamwork, and individual brilliance. Although the sport originated in England, it has grown far beyond its historical roots and is now followed passionately across countries such as India, Pakistan, Sri Lanka, Bangladesh, Nepal, Australia, New Zealand, South Africa, England, Afghanistan, and the West Indies. For millions of people, cricket is more than a sport. It is a source of identity, entertainment, national pride, community, and unforgettable memories.

At its simplest level, cricket is played between two teams, with each team normally consisting of eleven players. The basic objective is to score more runs than the opposing team. One team bats while the other team bowls and fields. The batting team attempts to score runs by hitting the ball, while the fielding team tries to restrict the scoring and dismiss the batters. However, beneath this simple description lies a highly complex game involving tactics, psychology, physical fitness, statistics, weather conditions, pitch characteristics, and constantly changing match situations.

"""
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0
)

result = splitter.split_text(text=text)
print(len(result))
print(result)