## Criar sprites a partir de imagens

cria uma nova entrada vazia ("new entry")
renomeia pra "S_START"

cria uma nova entrada vazia ("new entry")
renomeia pra "S_END"

move S_START pro inicio da lista de entradas
move S_END pra imediatamente abaixo de S_START

pra cada sprite:
    clica em "Import Files" (menu "Archive" -> "Import Files")
    o nome deve ter 4 letras, as letras seguinte (A1, A2A8, etc) devem seguir o padrão do monstro original
    seleciona o sprite
    botão direito, "Graphic", "Convert to..."
    no dropdown "Convert to..." escolher "Doom Gfx (Paletted)", clica em "Convert"
    
    dois cliques no sprite (ou clica e muda no menu à direita)
    perto de offset tem um dropdown list (aparentemente o padrão eh "Auto")
    clica nele e selecion "Sprite"
    clica no botão ao lado dele pra centralizar os sprite 
     - seta "Automatic Offsets" pra "Monster"
     - Ok
    salva e sai do menu do sprite
        
    (se o sprite for o msm, eu recomendo copiar e colar esse convertido e só mudar o nome)
    

## Criar um novo Decorate

cria uma nova entrada vazia ("new entry")
renomeia pra DECORATE (pode ser o que quiser?)
move DECORATE pro topo da lista (imediatamente antes de S_START)
seleciona ele na lista
botão direito, "View As" -> "View as Text"
herda do monstro original e muda só o sprite (no meu caso eu criei sprites com o nome "PBSP", por exemplo:

ACTOR Pbcacodemon : Cacodemon 15000
{
  States
  {
  Spawn:
    PBSP A 10 A_Look
    Loop
  }
}

(pra ver a lista de classes dos monstros ver na wiki do zdoom: https://zdoom.org/wiki/Classes:Cacodemon)

Edita o arquivo "SCRIPTS" e substitui todas chamadas à classe original para a nova classe (nesse exemplo seria "Pbcacodemon")
Botão direito no "SCRIPTS", "Script", "Compile ACS"


## Criar texturas a partir de imagens

Clica em "Texture Editor"
Se perguntar se quer criar arquivo de textura, dá Ok
no menu "Create Texture Definitions"
 - no "Format" seleciona "ZDoom (TEXTURES)"
 - e "Source" seleciona "Create New (Empty)"
Dois cliques no arquivo de textura
Clica no botão de importar imagem "New From File" (logo abaixo da lista de texturas, um botão com seta verde)
Importa a imagem
Salva e sai
Verifica se a nova imagem foi salva como uma textura entre tags "P_START" e "P_END"
move o TEXTURES ("Texture definitions") pra imediatamente antes de "P_START"
move todas as textures (do arquivo TEXTURES até P_END) para abaixo dos sprites (imediatamente abaixo de S_END)
 - se não tiver sprites, move pro topo


## Editar as texturas nos mapas

Dois cliques no mapa (MAP01) ou clica em "Map Editor"
Botão direito no que quiser alterar, "Change Texture" e escolhe as texturas novas
Salva e sai
