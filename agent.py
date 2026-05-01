import os
import sqlite3
import pandas as pd
import streamlit as st

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
model = init_chat_model("openai:gpt-4o-mini", temperature=0)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

DB_PATH = "vct.db"

system_prompt = """
You are Valorant Betting Model Assistant.

Your purpose is to help with a Valorant Champions Tour betting and analytics project.

You help with:
- team and player questions
- match and map questions
- event questions
- schedule questions
- database-related questions
- projections and betting model ideas
- analytics workflow help

Core behavior:
- Be clear, organized, and concise.
- Remember the earlier conversation in this same chat.
- Stay focused on Valorant esports, betting models, databases, workflows, and analytics.
- Keep answers simple, direct, and helpful.
- Do not end responses with generic chatbot phrases.
- End responses naturally and directly.

Grounding rules:
- Do not make up facts, table names, column names, tool results, or query results.
- If something is unknown, say so directly.
- Prefer tool-based answers over freeform reasoning whenever the question depends on live data.
- Never answer a live database question from memory.
- Never invent a schema.
- If document context is provided, use it only when relevant.

Tool rules:
- Use get_team_id when the user asks for a team_id.
- Use get_event_id when the user asks for an event_id.
- Use get_top_agents_for_player when the user asks which agents a player uses most.
- Use get_kill_avg_per_map when the user asks for a player's average kills per map.
- Use get_matches_on_date when the user asks what matches are on a certain date.
- Use list_tables and describe_table before writing SQL if the schema is uncertain.
- Use query_database for other live database questions.
- Only use read-only SQL.
- Only SELECT and PRAGMA queries are allowed.

Database safety rules:
- For live database questions, do not guess table names.
- If the needed table is uncertain, first use list_tables.
- If the likely table is found but columns are uncertain, use describe_table.
- Only after checking schema should you use query_database.
- If a query fails, explain the failure directly and briefly.
- Never claim there is no data unless a tool actually returned no data.

Follow-up question rules:
- Treat follow-up questions as referring to the same player, team, event, stat, and time context unless the user changes the context.
- Do not switch to a different lookup method in the middle of a follow-up sequence if a grounded tool already established the context.
- If a prior answer came from a tool, prefer continuing with tools that match that same entity and context.
- For follow-up stat questions, do not restate uncertainty if the needed context was already established in the previous turn.
- If the user asks a follow-up like "across how many maps?" or "how many kills over those maps?", resolve the pronouns and implied subject from the previous turn before answering.

Stat question rules:
- For exact player or team stats, prefer a dedicated tool if one exists.
- If no dedicated tool exists, inspect schema first, then use query_database.
- Do not answer exact stat questions with estimates.
- If the question asks for an average, return the average directly instead of listing raw rows unless the user asks for the breakdown.
- If the question asks for a leader, return the leader directly and include the supporting value when available.

Schedule rules:
- All schedule questions refer to 2026 unless the user clearly says another year.
- For schedule questions, return team names, region, and the stored match time.
- Do not return team IDs or match IDs unless the user explicitly asks for them.

Answer style:
- Give the answer first.
- Include the key number or result clearly.
- Keep the response short unless the user asks for more detail.
"""

def initialize_messages():
    return [{"role": "system", "content": system_prompt}]

@tool
def get_team_id(team_name: str) -> str:
    """
    Find the team_id for a given team name using the team CSV file.
    Input should be only the team name.
    Use this tool when the user asks for a team_id or asks which ID belongs to a team.
    """
    try:
        df = pd.read_csv("tool_files/SCHEMA_IDS_TEAMS_v1.0.csv")
    except Exception as e:
        return f"I could not open the team CSV file. Error: {e}"

    exact_match = df[df["team_name"].str.lower() == team_name.lower()]
    if not exact_match.empty:
        official_name = exact_match.iloc[0]["team_name"]
        team_id = exact_match.iloc[0]["team_id"]
        return f"The team_id for {official_name} is {team_id}."

    partial_match = df[df["team_name"].str.lower().str.contains(team_name.lower(), na=False)]
    if not partial_match.empty:
        official_name = partial_match.iloc[0]["team_name"]
        team_id = partial_match.iloc[0]["team_id"]
        return f"The team_id for {official_name} is {team_id}."

    return f"I could not find a team named '{team_name}' in the CSV file."

@tool
def get_event_id(event_name: str) -> str:
    """
    Find the event_id for a given event name using the event CSV file.
    Input should be only the event name.
    Use this tool when the user asks for an event_id or asks which ID belongs to an event.
    """
    try:
        df = pd.read_csv("tool_files/event_ids.csv")
    except Exception as e:
        return f"I could not open the event CSV file. Error: {e}"

    exact_match = df[df["event_name"].str.lower() == event_name.lower()]
    if not exact_match.empty:
        official_name = exact_match.iloc[0]["event_name"]
        event_id = exact_match.iloc[0]["event_id"]
        return f"The event_id for {official_name} is {event_id}."

    partial_match = df[df["event_name"].str.lower().str.contains(event_name.lower(), na=False)]
    if not partial_match.empty:
        official_name = partial_match.iloc[0]["event_name"]
        event_id = partial_match.iloc[0]["event_id"]
        return f"The event_id for {official_name} is {event_id}."

    return f"I could not find an event named '{event_name}' in the CSV file."

@tool
def list_tables(_: str = "") -> str:
    """
    List the tables in the SQLite database.
    Use this tool when the correct table is uncertain.
    """
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        query = """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            return "No tables were found in the database."

        return df.to_string(index=False)

    except Exception as e:
        return f"Schema lookup error: {e}"

@tool
def describe_table(table_name: str) -> str:
    """
    Show the columns for one table in the SQLite database.
    Input should be only the table name.
    Use this tool when you know the likely table but need to inspect its columns.
    """
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
    if not table_name or any(ch not in allowed_chars for ch in table_name):
        return "Invalid table name."

    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        query = f"PRAGMA table_info({table_name})"
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            return f"No schema information found for table '{table_name}'."

        return df.to_string(index=False)

    except Exception as e:
        return f"Schema lookup error: {e}"

@tool
def get_top_agents_for_player(player_name: str) -> str:
    """
    Return the 3 most common agents played by a player from the live database.

    Input should be only the player name.
    Use this tool when the user asks what agents a player plays most often
    or asks for the player's most common agents.
    """
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)

        exact_sql = """
        SELECT p.player_name, a.agent_name, COUNT(*) AS maps_played
        FROM player_map_agents pma
        JOIN players p ON p.player_id = pma.player_id
        JOIN agents a ON a.agent_id = pma.agent_id
        WHERE LOWER(p.player_name) = LOWER(?)
        GROUP BY p.player_name, a.agent_name
        ORDER BY maps_played DESC, a.agent_name
        LIMIT 3
        """
        exact_df = pd.read_sql_query(exact_sql, conn, params=[player_name])

        if not exact_df.empty:
            conn.close()
            lines = [f"The 3 most common agents for {exact_df.iloc[0]['player_name']} are:"]
            for _, row in exact_df.iterrows():
                lines.append(f"{row['agent_name']} - {int(row['maps_played'])} maps")
            return "\n".join(lines)

        partial_sql = """
        SELECT p.player_name, a.agent_name, COUNT(*) AS maps_played
        FROM player_map_agents pma
        JOIN players p ON p.player_id = pma.player_id
        JOIN agents a ON a.agent_id = pma.agent_id
        WHERE LOWER(p.player_name) LIKE LOWER(?)
        GROUP BY p.player_name, a.agent_name
        ORDER BY maps_played DESC, a.agent_name
        """
        partial_df = pd.read_sql_query(partial_sql, conn, params=[f"%{player_name}%"])
        conn.close()

        if not partial_df.empty:
            matched_name = partial_df.iloc[0]["player_name"]
            player_only = partial_df[partial_df["player_name"] == matched_name].head(3)

            lines = [f"The 3 most common agents for {matched_name} are:"]
            for _, row in player_only.iterrows():
                lines.append(f"{row['agent_name']} - {int(row['maps_played'])} maps")
            return "\n".join(lines)

        return f"I could not find agent usage data for player '{player_name}'."

    except Exception as e:
        return f"Agent usage lookup error: {e}"

@tool
def get_kill_avg_per_map(player_name: str) -> str:
    """
    Return the average kills per map for a player from the live database.

    Input should be only the player name.
    Use this tool when the user asks for a player's kill average per map,
    average kills, or kills per map.
    """
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)

        exact_sql = """
        SELECT
            p.player_name,
            COUNT(*) AS maps_played,
            SUM(pms.kills) AS total_kills,
            AVG(pms.kills * 1.0) AS avg_kills_per_map
        FROM player_map_stats pms
        JOIN players p ON p.player_id = pms.player_id
        WHERE LOWER(p.player_name) = LOWER(?)
        GROUP BY p.player_name
        """
        exact_df = pd.read_sql_query(exact_sql, conn, params=[player_name])

        if not exact_df.empty:
            conn.close()
            row = exact_df.iloc[0]
            return (
                f"{row['player_name']} averages "
                f"{row['avg_kills_per_map']:.2f} kills per map "
                f"across {int(row['maps_played'])} maps "
                f"({int(row['total_kills'])} total kills)."
            )

        partial_sql = """
        SELECT
            p.player_name,
            COUNT(*) AS maps_played,
            SUM(pms.kills) AS total_kills,
            AVG(pms.kills * 1.0) AS avg_kills_per_map
        FROM player_map_stats pms
        JOIN players p ON p.player_id = pms.player_id
        WHERE LOWER(p.player_name) LIKE LOWER(?)
        GROUP BY p.player_name
        ORDER BY maps_played DESC, p.player_name
        """
        partial_df = pd.read_sql_query(partial_sql, conn, params=[f"%{player_name}%"])
        conn.close()

        if not partial_df.empty:
            row = partial_df.iloc[0]
            return (
                f"{row['player_name']} averages "
                f"{row['avg_kills_per_map']:.2f} kills per map "
                f"across {int(row['maps_played'])} maps "
                f"({int(row['total_kills'])} total kills)."
            )

        return f"I could not find kill data for player '{player_name}'."

    except Exception as e:
        return f"Kill average lookup error: {e}"

@tool
def get_matches_on_date(match_date: str) -> str:
    """
    Return all matches on a given date using the live database.

    Input should be a date in YYYY-MM-DD format.
    Output should show:
    Region - Team 1 vs Team 2 - stored time

    Use this tool when the user asks what matches are on a certain date.
    """
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)

        sql = """
        SELECT
            m.region,
            t1.team_name AS team1_name,
            t2.team_name AS team2_name,
            m.match_date
        FROM matches m
        JOIN teams t1
            ON t1.team_id = m.team1_id
        JOIN teams t2
            ON t2.team_id = m.team2_id
        WHERE substr(m.match_date, 1, 10) = ?
        ORDER BY m.match_date, m.match_id
        """

        df = pd.read_sql_query(sql, conn, params=[match_date])
        conn.close()

        if df.empty:
            return f"No matches were found on {match_date}."

        lines = [f"Matches on {match_date}:"]
        for _, row in df.iterrows():
            stored_time = str(row["match_date"])[10:].strip()
            if stored_time:
                lines.append(
                    f"{row['region']} - {row['team1_name']} vs {row['team2_name']} - {stored_time}"
                )
            else:
                lines.append(
                    f"{row['region']} - {row['team1_name']} vs {row['team2_name']}"
                )

        return "\n".join(lines)

    except Exception as e:
        return f"Schedule lookup error: {e}"

@tool
def query_database(sql_query: str) -> str:
    """
    Run a read-only SQL query on the Valorant SQLite database.

    Allowed queries:
    - SELECT
    - PRAGMA

    This tool should be used for live database questions that are not simple
    team_id or event_id CSV lookups.

    Output is limited to the first 20 rows.
    """
    query = sql_query.strip()
    query_upper = query.upper()

    if not (query_upper.startswith("SELECT") or query_upper.startswith("PRAGMA")):
        return "Only SELECT and PRAGMA queries are allowed."

    blocked_words = [
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
        "REPLACE", "TRUNCATE", "ATTACH", "DETACH", "VACUUM"
    ]

    for word in blocked_words:
        if word in query_upper:
            return f"Blocked query. {word} is not allowed."

    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)

        if query_upper.startswith("SELECT"):
            limited_query = f"SELECT * FROM ({query.rstrip(';')}) LIMIT 20"
            df = pd.read_sql_query(limited_query, conn)
        else:
            df = pd.read_sql_query(query, conn).head(20)

        conn.close()

        if df.empty:
            return "Query ran successfully but returned no rows."

        return df.to_string(index=False)

    except Exception as e:
        return f"Query error: {e}"

agent = create_agent(
    model=model,
    tools=[
        get_team_id,
        get_event_id,
        list_tables,
        describe_table,
        get_top_agents_for_player,
        get_kill_avg_per_map,
        get_matches_on_date,
        query_database
    ],
    system_prompt=system_prompt
)

def get_response(messages, user_input):
    docs = vectorstore.similarity_search(user_input, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    augmented_input = f"""
Use this document context if it is relevant.

Context:
{context}

Question:
{user_input}
"""

    messages.append({"role": "user", "content": user_input})

    temp_messages = messages[:-1] + [{"role": "user", "content": augmented_input}]

    result = agent.invoke({"messages": temp_messages})
    assistant_reply = result["messages"][-1].content

    messages.append({"role": "assistant", "content": assistant_reply})

    return assistant_reply, messages
