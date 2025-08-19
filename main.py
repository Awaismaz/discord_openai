import os
import discord
from discord import app_commands
from dotenv import load_dotenv
from services.openai_chat import chat_fast
load_dotenv(override=True)
from services.ratelimit import allow as rl_allow, reset_user as rl_reset
from services.openai_coach import coach_answer, reset_user_thread
from services import logger
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
CHAT_CHANNEL = os.getenv("CHAT_CHANNEL", "chat")
COACH_CHANNEL = os.getenv("COACH_CHANNEL", "coach")

intents = discord.Intents.default()
intents.message_content = False  # we’ll rely on slash commands

class Bot(discord.Client):
    def __init__(self):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        # Sync commands to guild(s); global sync may take up to 1h, so for dev use guild sync:
        # Replace GUILD_ID with your test server ID for instant sync, or comment out for global.
        # guild = discord.Object(id=YOUR_GUILD_ID)
        # await self.tree.sync(guild=guild)
        await self.tree.sync()

client = Bot()

def in_allowed_channel(interaction: discord.Interaction, allowed: str):
    return interaction.channel and interaction.channel.name == allowed

@client.tree.command(name="health", description="Bot health check")
async def health(interaction: discord.Interaction):
    await interaction.response.send_message("✅ Online (core skeleton running).")

@client.tree.command(name="chat", description="Chat mode (fast Q&A)")
async def chat_cmd(interaction: discord.Interaction, prompt: str):
    if not in_allowed_channel(interaction, CHAT_CHANNEL):
        return await interaction.response.send_message(
            f"Please use #{CHAT_CHANNEL} for /chat.", ephemeral=True
        )
    ok, remaining = rl_allow(str(interaction.user.id), "chat")
    if not ok:
        return await interaction.response.send_message(
            "⏳ Rate limit reached (chat). Please retry in a minute.", ephemeral=True
        )
    await interaction.response.defer(thinking=True, ephemeral=False)
    reply = chat_fast(prompt, user_id=str(interaction.user.id))
    await interaction.followup.send(f"🗨️ **Chat:** {reply}")

# add imports
from typing import Optional
from services.openai_coach import coach_answer, reset_user_thread

@client.tree.command(name="coach", description="Coach mode (PDF/TXT + citations)")
@app_commands.describe(question="Your question about the file or topic",
                       file="Optional file: PDF/TXT (<=15MB)")
async def coach_cmd(interaction: discord.Interaction,
                    question: Optional[str] = None,
                    file: Optional[discord.Attachment] = None):
    if not in_allowed_channel(interaction, COACH_CHANNEL):
        return await interaction.response.send_message(
            f"Please use #{COACH_CHANNEL} for /coach.", ephemeral=True
        )
    ok, remaining = rl_allow(str(interaction.user.id), "coach")
    if not ok:
        return await interaction.response.send_message(
            "⏳ Rate limit reached (coach). Please retry in a minute.", ephemeral=True
        )
    # Prepare attachment metadata for service
    attach = None
    if file:
        attach = {
            "url": file.url,
            "filename": file.filename,
            "content_type": file.content_type or "application/octet-stream",
            "size": file.size,
        }

    await interaction.response.defer(thinking=True)
    try:
        reply = await coach_answer(user_id=str(interaction.user.id),
                                question=question, attachment=attach)
    except Exception as e:
        logger.exception("Coach mode failed for user %s", interaction.user.id)
        reply = "⚠️ Sorry, I couldn’t process your request. Please try again."
    await interaction.followup.send(f"🎓 **Coach:** {reply}")


@client.tree.command(name="reset", description="Reset your session context")
@app_commands.describe(mode="Which context to reset: chat/coach/all")
async def reset_cmd(interaction: discord.Interaction, mode: Optional[str] = "coach"):
    m = (mode or "coach").lower()
    uid = str(interaction.user.id)
    if m in ("coach", "all"):
        reset_user_thread(uid)  # clear Assistants thread
        rl_reset(uid, "coach")
    if m in ("chat", "all"):
        # chat is stateless in Phase 1; still clear rate-bucket
        rl_reset(uid, "chat")
    await interaction.response.send_message(f"♻️ Reset completed for `{m}`.")


@client.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    logger.exception("Error in command %s", interaction.command.name if interaction.command else "unknown")
    # Graceful reply
    if interaction.response.is_done():
        await interaction.followup.send("⚠️ Sorry, something went wrong. Please try again.", ephemeral=True)
    else:
        await interaction.response.send_message("⚠️ Sorry, something went wrong. Please try again.", ephemeral=True)


client.run(TOKEN)
