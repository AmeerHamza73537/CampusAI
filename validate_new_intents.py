from chatbot import ChatbotEngine

bot = ChatbotEngine()
queries = [
    'Does the university have a library?',
    'How many departments does the university have?',
    'Is there on-campus accommodation available for female students specifically?',
    'Does the university have a disability support or accessibility office?',
    'What are the library operating hours during exam season?',
    'When are the final exams usually held?',
    'How can I check my results?',
    'What is the policy for missing an exam due to a medical emergency?',
    'How many times can I repeat a failed course?',
    'If I have a grading dispute, what is the process to raise it?',
    'What is your favorite movie?',
    'Can you tell me how to fix my car?',
    'Do you know how to bake a cake?'
]
for q in queries:
    r = bot.predict(q)
    print('Q:', q)
    print('Intent:', r['intent'])
    print('Resp:', r['response'])
    print('---')
