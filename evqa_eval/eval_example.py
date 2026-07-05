import evaluation_utils

question = 'Where does this hotel derive its name from?'
question_type = 'automatic'

# semantically correct answer.
score1 = evaluation_utils.evaluate_example(
    question,
    reference_list=['the Menger family'],
    candidate='William and Mary Menger',
    question_type=question_type)

# semantically incorrect answer.
score2 = evaluation_utils.evaluate_example(
    question,
    reference_list=['the Menger family'],
    candidate='the Baker family',
    question_type=question_type)

# score1 is 1.0, score2 is 0.0.
print(f'{score1=}, {score2=}')