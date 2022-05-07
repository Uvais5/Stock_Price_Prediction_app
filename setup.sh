mkdir -p ~/.streamlit/
echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless =Itrue\n\
          
" > ~/.streamlit/config.toml