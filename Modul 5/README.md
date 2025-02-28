# Modul 5: Reinforcement Learning

## Daftar Isi
- [Modul 5: Reinforcement Learning](#modul-5-reinforcement-learning)
    - [Daftar Isi](#daftar-isi)
    - [Terminologi](#terminologi)
    - [Pengenalan](#pengenalan)
    - [Pendekatan Reinforcement Learning](#pendekatan-reinforcement-learning)
    - [Algoritma](#algoritma)
        - [Q-Learning](#q-learning)
        - [NEAT Algorithm](#neat-algorithm)


## Terminologi
- `Agent`: mempunyai tugas untuk mencapai tujuan (goal)
- `Environment`: memberikan feedback terhadap aksi yang dilakukan Agen
- `Current State` (s): kondisi atau situasi saat ini berdasarkan perspektif Agen
- `Next State` (s'): kondisi atau situasi berikutnya setelah Agen melakukan aksi
- `Goal`: tujuan yang ingin dicapai oleh Agen
- `Action` (a): aksi yang akan dipilih Agen untuk mencapai tujuan
- `Policy` (\( \pi \)): strategi / kebijakan yang digunakan Agen untuk memilih aksi
- `Reward` (R): sebuah nilai untuk mengukur keberhasilan aksi dari Agen
- `Penalty`: sebuah nilai untuk mengukur kegagalan aksi dari Agen

## Pengenalan
Reinforcement Learning (RL) adalah sebuah teknik dalam *machine learning* yang mempelajari bagaimana agen harus bertindak dalam sebuah lingkungan agar mendapatkan reward yang maksimal. RL bekerja berdasarkan konsep **trial and error** sehingga agen mengeksplorasi berbagai tindakan, menerima feedback berupa reward atau penalty, dan menyesuaikan strateginya untuk mencapai tujuan yang optimal.

Faktanya, RL banyak digunakan dalam berbagai aplikasi seperti *game*, *robotics*, *recommendation systems*, *search engines*, dan lainnya.

> Untuk modul ini, kita akan mengimplemtasikan RL pada *game* sederhana 🎮

## Pendekatan Reinforcement Learning
Ada beberapa pendekatan yang dapat digunakan dalam RL, diantaranya:
1. **Value-Based**: Menentukan policy secara tidak langsung dengan **mempelajari fungsi nilai aksi** \( Q(s, a) \) dan memilih aksi dengan nilai tertinggi.  
   \[
   Q(s, a) \leftarrow Q(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right)
   \]

- **\( Q(s, a) \)** = Seberapa baik Agen dalam mengambil aksi (\( a \)) yang ada pada suatu state (\( s \)).  
- **\( \alpha \)** = *Learning rate* pada fungsi nilai.
- **\( r \)** = Reward yang diterima setelah melakukan aksi (\( a \)).  
- **\( \gamma \)** = *Discount factor* yang menentukan seberapa jauh Agen mempertimbangkan reward masa depan.  
- **\( \max_{a'} Q(s', a') \)** = Nilai terbaik yang bisa diperoleh dari state selanjutnya (\( s' \)).  

2. **Policy-Based**: **Mempelajari policy \( \pi(a | s) \)** tanpa perlu fungsi nilai, dengan menggunakan metode optimasi seperti *gradient ascent*.  
   \[
   \nabla J(\theta) = \mathbb{E} \left[ \nabla_{\theta} \log \pi_{\theta} (a | s) R \right]
   \]
- **\( J(\theta) \)** = Fungsi objektif berupa reward yang ingin dimaksimalkan.  
- **\( \theta \)** = Parameter dari policy (\( \pi \))  
- **\( \mathbb{E} [\cdot] \)** = Ekspektasi rata-rata dari sampel yang diperoleh selama eksplorasi.
- **\( \nabla_{\theta} \log \pi_{\theta} (a | s) \)** = Gradien dari logaritma policy.
- **\( \pi_{\theta} (a | s) \)** = Probabilitas memilih aksi (\( a \)) dalam state (\( s \)).  
- **\( R \)** = Reward total yang diperoleh setelah mengambil aksi \( a \). 

> Referensi: [REINFORCE Algorithm](https://medium.com/intro-to-artificial-intelligence/reinforce-a-policy-gradient-based-reinforcement-learning-algorithm-84bde440c816)

3. **Model-Based** membangun **model transisi lingkungan** \( P(s' | s, a) \) untuk memprediksi keadaan berikutnya sebelum agen mengambil keputusan.  
   \[
   s' \sim P(s' | s, a)
   \]
- **\( s' \)** = Next state setelah Agen mengambil aksi (\( a \)) pada state (\( s \)).  
- **\( \sim \)** = \( s' \) diambil secara acak dari distribusi probabilitas \( P(s' | s, a) \).  
- **\( P(s' | s, a) \)** = Probabilitas transisi dari state (\( s \)) ke state (\( s' \)) setelah mengambil aksi (\( a \)).  

## Algoritma
### Q-Learning
Q-Learning adalah salah satu algoritma dalam RL yang termasuk dalam kategori **Value-Based**. Algoritma ini mempelajari fungsi nilai aksi \( Q(s, a) \) untuk memaksimalkan reward yang diperoleh Agen.

**Contoh Implementasi**
Untuk implementasinya, bisa di cek pada kode [taxi.py](/Modul%205/Q_Learning/taxi.py). Kode ini merupakan implementasi Q-Learning pada *game* Taxi-v3 dari OpenAI Gym. Goal dari kode ini adalah melatih Agen untuk mengambil penumpang dan mengantarkannya ke tujuan dengan efisien.

![Taxi-v3](/Modul%205/assets/taxi_image.png)

```python
for i in range(training_episodes):
    state, _ = env.reset()
    done = False
    penalties, reward = 0, 0
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = numpy.argmax(q_table[state])

        next_state, reward, done, _, _ = env.step(action) 
        
        old_value = q_table[state, action]
        next_max = numpy.max(q_table[next_state])

        # Update q-value
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max) # Sesuai rumus Q-Learning
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        
    if i % 100 == 0:
        print(f"Episode: {i}")

print("Training finished.\n")
```

### NEAT Algorithm
NEAT (NeuroEvolution of Augmenting Topologies) adalah algoritma yang menggabungkan *genetic algorithm* dan *neural network* untuk menemukan arsitektur jaringan saraf yang optimal. Well, sebenarnya NEAT bukanlah algoritma RL, namun algoritma ini dapat melatih Agen dalam sebuah lingkungan RL.

> Referensi: [NEAT Algorithm](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)

**Contoh Implementasi**
Untuk implementasinya, bisa di cek pada kode [neat_car.py](/Modul%205/NEAT/neat_car.py). Goal dari kode ini adalah melatih mobil untuk mengelilingi lintasan tanpa menabrak dinding.

![NEAT](//Modul%205/assets/neat_car_image.png)

```python
def run_car(genomes, config):
    nets = []
    cars = []

    for id, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        cars.append(Car())

    # Init game
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.DOUBLEBUF)
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 70)
    font = pygame.font.SysFont("Arial", 30)
    map = pygame.image.load(gambar_peta)

    # Main loop
    global generation
    generation += 1
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)


        # Input data and get result from network
        for index, car in enumerate(cars):
            output = nets[index].activate(car.get_data())
            i = output.index(max(output))
            if i == 0:
                car.angle += 10
            else:
                car.angle -= 10

        # Update car and fitness
        remain_cars = 0
        for i, car in enumerate(cars):
            if car.get_alive():
                remain_cars += 1
                car.update(map)
                genomes[i][1].fitness += car.get_reward()

        # check
        if remain_cars == 0:
            break

        # Drawing
        screen.blit(map, (0, 0))
        for car in cars:
            if car.get_alive():
                car.draw(screen)

        text = generation_font.render("Generation : " + str(generation), True, (255, 255, 0))
        text_rect = text.get_rect()
        text_rect.center = (screen_width/2, 100)
        screen.blit(text, text_rect)

        text = font.render("remain cars : " + str(remain_cars), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (screen_width/2, 200)
        screen.blit(text, text_rect)

        pygame.display.update()
        clock.tick(60)
```